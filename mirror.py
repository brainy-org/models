# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     # The docling and RapidOCR that brainy-worker locks: the bundle holds
#     # exactly the files this docling resolves (folders, RapidOCR's per-
#     # language recognisers). Bump them with the worker's uv.lock.
#     "docling-slim==2.137.0",
#     "rapidocr==3.10.0",
#     "huggingface-hub==1.33.0",
# ]
# ///
"""Mirror what the brainy-org services download from third parties.

    uv run mirror.py build [BUNDLE...]     fetch into .work/, verify, write NOTICE.md
    uv run mirror.py push [BUNDLE...]      build, push to ghcr.io/brainy-org, update the lock
    uv run mirror.py images                copy the third-party images to ghcr.io/brainy-org/mirror
    uv run mirror.py vendor                push the build inputs image (ffmpeg, libparakeet)

Every source is a commit (or a file with its sha256), never a branch or a
tag: what is mirrored is what was reviewed. A bundle is an OCI artifact, one
layer per file titled with its path; consumers pull it by digest, which
ghcr.io cannot change under them. models.lock.json records, per bundle, the
digest and every file's sha256 and size: the consumers copy the entry they
use (README.md, "Consumers").

Needs `oras` and a login to ghcr.io with write:packages
(`gh auth token | oras login ghcr.io -u <user> --password-stdin`, and the
same for `docker login` before `vendor`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".work"
LOCK = ROOT / "models.lock.json"
REGISTRY = "ghcr.io/brainy-org"
SOURCE = "https://github.com/brainy-org/models"
ARTIFACT_TYPE = "application/vnd.brainy-org.model-bundle.v1"


@dataclass(frozen=True)
class HfSource:
    repo: str
    commit: str
    licence: str
    # Files of the snapshot, as huggingface_hub patterns; None: all of them.
    allow: list[str] | None = None
    ignore: list[str] | None = None
    # Directory inside the bundle; "" for its root.
    into: str = ""
    note: str = ""


@dataclass(frozen=True)
class Bundle:
    name: str
    # The tag next to the digest: names what the bundle is for a reader.
    tag: str
    used_by: list[str]
    sources: list[HfSource] = field(default_factory=list)
    extra: Callable[[Path], list[dict[str, str]]] | None = None
    description: str = ""


def _rapidocr(directory: Path) -> list[dict[str, str]]:
    """Every RapidOCR checkpoint docling may load on onnxruntime, under
    RapidOcr/ as docling's artifacts_path expects it: the PP-OCRv6 small
    det/rec (Latin, Chinese, Japanese...), the v4 classifier, and the v5/v4
    mobile recogniser of each other script (a job's `iso:<lang>`). Checked
    against the SHA256 that rapidocr's default_models.yaml lists."""
    import yaml
    from docling.models.stages.ocr import rapid_ocr_model as r
    from rapidocr.utils.typings import EngineType

    expected: dict[str, str] = {}
    table = yaml.safe_load(
        (Path(sys.modules["rapidocr"].__file__).parent / "default_models.yaml").read_text()
    )

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "model_dir" in node and "SHA256" in node:
                expected[node["model_dir"]] = node["SHA256"]
            for value in node.values():
                walk(value)

    walk(table.get("onnxruntime", {}))
    target = directory / r.RapidOcrModel._model_repo_folder
    target.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, str] = {}
    for code in sorted(r._rapidocr_vocabulary("onnxruntime")):
        # Raises for a code it cannot serve: a bundle missing a language
        # would fail that language's jobs offline, so the build stops.
        spec = r._resolve_rapidocr(code, "onnxruntime")
        artifacts = r._rapidocr_artifacts(
            target, EngineType.ONNXRUNTIME, spec.ppocr_version, spec.rapidocr_code
        )
        for artifact in artifacts.values():
            for destination, url in artifact.files.items():
                fetched[url] = str(destination.relative_to(directory))
                if destination.exists():
                    continue
                print(f"  rapidocr {code}: {url}", flush=True)
                with urllib.request.urlopen(url) as response, destination.open("wb") as out:
                    shutil.copyfileobj(response, out)
    for url, relative in fetched.items():
        want = expected.get(url)
        if want is None:
            sys.exit(f"rapidocr: {url} has no SHA256 in default_models.yaml")
        got = _sha256(directory / relative)
        if got != want:
            sys.exit(f"rapidocr: {relative} is {got}, default_models.yaml says {want}")
    return [
        {"url": url, "path": relative, "licence": "Apache-2.0 (PaddleOCR, RapidOCR)"}
        for url, relative in sorted(fetched.items())
    ]


def _release(
    files: dict[str, str], base: str, licence: str
) -> Callable[[Path], list[dict[str, str]]]:
    """Files of a release, each checked against the sha256 it was reviewed with."""

    def fetch(directory: Path) -> list[dict[str, str]]:
        notes = []
        for name, sha256 in files.items():
            path = directory / name
            url = f"{base}/{name}"
            if not path.exists():
                print(f"  {url}", flush=True)
                with urllib.request.urlopen(url) as response, path.open("wb") as out:
                    shutil.copyfileobj(response, out)
            if _sha256(path) != sha256:
                sys.exit(f"{name} does not match its sha256")
            notes.append({"url": url, "path": name, "licence": licence})
        return notes

    return fetch


def _bundle_component(
    url: str, sha256: str, component: str, name: str, source_sha256: str, licence: str
) -> Callable[[Path], list[dict[str, str]]]:
    """One model of a parakeet.cpp bundle GGUF, written back as the
    standalone GGUF it was packed from: the bundle's keys and tensors under
    `<component>.`, without the prefix, in order, with the same data. The
    bundle names that file's sha256 (`parakeet.bundle.<component>.
    source_sha256`), and the result must have it: the file mirrored is the
    upstream file byte for byte, without the bundle's other models."""

    def fetch(directory: Path) -> list[dict[str, str]]:
        cache = WORK / "downloads"
        cache.mkdir(parents=True, exist_ok=True)
        packed = cache / url.rsplit("/", 1)[1]
        if not packed.exists() or _sha256(packed) != sha256:
            print(f"  {url}", flush=True)
            with urllib.request.urlopen(url) as response, packed.open("wb") as out:
                shutil.copyfileobj(response, out)
        if _sha256(packed) != sha256:
            sys.exit(f"{packed.name} does not match its sha256")
        out = directory / name
        _extract_component(packed, component, out)
        if _sha256(out) != source_sha256:
            sys.exit(f"{name}, extracted from {packed.name}, is not the file the bundle names")
        return [{"url": f"{url} (component `{component}`)", "path": name, "licence": licence}]

    return fetch


def _extract_component(bundle: Path, component: str, out: Path) -> None:
    import struct

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with bundle.open("rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{bundle.name} is not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def string() -> bytes:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n)

        def value(kind: int) -> bytes:
            # The value's bytes as stored, to be written back unchanged.
            if kind in sizes:
                return f.read(sizes[kind])
            if kind == 8:
                s = string()
                return struct.pack("<Q", len(s)) + s
            if kind == 9:
                (item,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                return struct.pack("<IQ", item, n) + b"".join(value(item) for _ in range(n))
            sys.exit(f"{bundle.name}: unknown GGUF value type {kind}")

        prefix = component.encode() + b"."
        keys, alignment = [], 32
        for _ in range(n_kv):
            key = string()
            (kind,) = struct.unpack("<I", f.read(4))
            raw = value(kind)
            if key.startswith(prefix):
                keys.append((key[len(prefix) :], kind, raw))
                if key[len(prefix) :] == b"general.alignment":
                    (alignment,) = struct.unpack("<I", raw)
        tensors = []
        for _ in range(n_tensors):
            name = string()
            (dims,) = struct.unpack("<I", f.read(4))
            shape = f.read(8 * dims)
            kind, offset = struct.unpack("<IQ", f.read(12))
            if name.startswith(prefix):
                tensors.append((name[len(prefix) :], dims, shape, kind, offset))
        if not tensors:
            sys.exit(f"{bundle.name} has no component {component}")
        data = (f.tell() + 31) // 32 * 32
        first = min(t[4] for t in tensors)
        last = max(tensors, key=lambda t: t[4])
        elements = 1
        for i in range(last[1]):
            elements *= struct.unpack("<Q", last[2][8 * i : 8 * i + 8])[0]
        type_size = {0: 4, 1: 2}.get(last[3])
        if type_size is None:
            sys.exit(f"{bundle.name}: the last tensor of {component} is not F32 or F16")
        header = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(keys)))
        for key, kind, raw in keys:
            header += struct.pack("<Q", len(key)) + key + struct.pack("<I", kind) + raw
        for name, dims, shape, kind, offset in tensors:
            header += struct.pack("<Q", len(name)) + name + struct.pack("<I", dims) + shape
            header += struct.pack("<IQ", kind, offset - first)
        header += b"\0" * (-len(header) % alignment)
        f.seek(data + first)
        body = f.read(last[4] - first + elements * type_size)
    out.write_bytes(bytes(header) + body)


# libparakeet v0.6.1, the builds brainy-worker's image and logbook load; the
# sha256 are the release's own (GitHub's asset digests, 2026-10-24).
PARAKEET_RELEASE = {
    "parakeet-v0.6.1-lib-linux-cpu-x64.tar.gz": "5a2b439e83d1b6ab66c55bcfc48751ee48e53727d5b3ea33a6bc0e8626f1397c",
    "parakeet-v0.6.1-lib-linux-cpu-arm64.tar.gz": "d9e56bab07f87b6c05593a039f7cfe5b6033995da80f21721f167bfb41698c42",
    "parakeet-v0.6.1-lib-linux-cuda-x64.tar.gz": "f717526c49d56cd4691390ca4c8e521a27eec698a93f3897bf3817ceb744378a",
    "parakeet-v0.6.1-lib-macos-metal-arm64.tar.gz": "1847118600359355c899619e263275d83b59f81dcc34fcca6ba4e6b9aac67db0",
    "parakeet-v0.6.1-lib-macos-cpu-x64.tar.gz": "4575814a5abb2d8d78b657750a3418cf2178765aa8f2f4ebfd692bf78fda9e7d",
}


BUNDLES: dict[str, Bundle] = {
    bundle.name: bundle
    for bundle in (
        Bundle(
            name="docling",
            tag="docling-2.137.0",
            used_by=["brainy-worker (docling's artifacts_path)"],
            description="docling layout, TableFormer, CodeFormula and RapidOCR models",
            sources=[
                # docling 2.137 asks for `main`; this is main on 2026-10-24.
                HfSource(
                    "docling-project/docling-layout-heron",
                    "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
                    "Apache-2.0",
                    into="docling-project--docling-layout-heron",
                    ignore=["*.png"],
                ),
                # The tag v2.3.0 docling asks for; ACCURATE only (the worker's mode).
                HfSource(
                    "docling-project/docling-models",
                    "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
                    "CDLA-Permissive-2.0",
                    into="docling-project--docling-models",
                    allow=["README.md", "config.json", "model_artifacts/tableformer/accurate/*"],
                ),
                HfSource(
                    "docling-project/CodeFormulaV2",
                    "ecedbe111d15c2dc60bfd4a823cbe80127b58af4",
                    "CDLA-Permissive-2.0",
                    into="docling-project--CodeFormulaV2",
                ),
            ],
            extra=_rapidocr,
        ),
        Bundle(
            name="parakeet",
            tag="741158ae",
            used_by=["brainy-worker (recordings)", "logbook"],
            description="parakeet.cpp GGUF models: ASR, VAD, diarization",
            sources=[
                HfSource(
                    "mudler/parakeet-cpp-gguf",
                    "741158ae71e64ef5c89385862c18f777d07a97a1",
                    "CC-BY-4.0 (nvidia/parakeet-tdt-0.6b-v3), OpenMDW-1.1 "
                    "(nvidia/Nemotron-3-Diarization), MIT (Silero VAD)",
                    allow=[
                        "README.md",
                        "tdt-0.6b-v3-q8_0.gguf",
                        "silero-vad-f16.gguf",
                        "nemotron-3-diarization-q8_0.gguf",
                    ],
                    note="Parakeet TDT 0.6B v3 by NVIDIA, CC-BY-4.0, converted to "
                    "GGUF by mudler (parakeet.cpp).",
                ),
            ],
        ),
        Bundle(
            name="parakeet-voice",
            tag="741158ae",
            used_by=["logbook (final pass: more than eight speakers)"],
            description="WeSpeaker ResNet34-LM speaker encoder, as parakeet.cpp loads it (GGUF)",
            extra=_bundle_component(
                "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/"
                "741158ae71e64ef5c89385862c18f777d07a97a1/parakeet-bundle-small.gguf",
                "5f2c6697eb08d4d67e858692d4223a9d911aacc51ae16cf2aee9e760f8a2269a",
                "voice",
                "wespeaker-resnet34-lm.gguf",
                "72040372494eafec299836bc1977cfc13c603cb486674ed59b0f4c03758d29da",
                "CC-BY-4.0: WeSpeaker ResNet34-LM (large-margin fine-tuned, trained on "
                "VoxCeleb2 dev) by the WeSpeaker project (https://github.com/wenet-e2e/wespeaker), "
                "converted from its ONNX export to GGUF by mudler (voice-detect.cpp, F32)",
            ),
        ),
        Bundle(
            name="libparakeet",
            tag="v0.6.1",
            used_by=["logbook (scripts/server.sh)"],
            description="libparakeet v0.6.1 release builds (parakeet.cpp's C API)",
            extra=_release(
                PARAKEET_RELEASE,
                "https://github.com/mudler/parakeet.cpp/releases/download/v0.6.1",
                "MIT (parakeet.cpp, ggml); the CUDA build carries NVIDIA's "
                "redistributable cudart and cuBLAS",
            ),
        ),
        Bundle(
            name="qwen3-embedding-0.6b",
            tag="3d106eab",
            used_by=["brainy (TEI)", "galley (TEI)"],
            description="Qwen3-Embedding-0.6B with its ONNX export, for TEI",
            sources=[
                # refs/pr/27 (the ONNX export) on 2026-10-24. The PyTorch
                # weights are left out: TEI runs the ONNX ones, as it did
                # when it downloaded them itself.
                HfSource(
                    "Qwen/Qwen3-Embedding-0.6B",
                    "3d106eabb5535a84de3ae88f45887a78259b52de",
                    "Apache-2.0",
                    ignore=["model.safetensors", ".gitattributes"],
                ),
            ],
        ),
    )
}

# Third-party images the charts and compose files run, copied as they are.
IMAGES = {
    "ghcr.io/huggingface/text-embeddings-inference:cpu-1.9": "mirror/text-embeddings-inference:cpu-1.9",
    "ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9": "mirror/text-embeddings-inference:cpu-arm64-1.9",
    "ghcr.io/oras-project/oras:v1.3.0": "mirror/oras:v1.3.0",
}

# Mirrored images whose platform builds are published under separate tags,
# joined into one multi-platform tag: the compose files run it on amd64 and
# arm64 hosts alike. Each platform's manifest keeps its digest.
MULTIARCH = {
    "mirror/text-embeddings-inference:cpu-1.9-multiarch": [
        "mirror/text-embeddings-inference:cpu-1.9",
        "mirror/text-embeddings-inference:cpu-arm64-1.9",
    ],
}

# The build inputs brainy-worker's Dockerfile downloads, checked against the
# same sha256 its Dockerfile pins.
VENDOR_TAG = "brainy-worker-build:2026-10-24"
VENDOR_FILES = {
    "ffmpeg-8.1.3.tar.xz": (
        "https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz",
        "7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3",
    ),
    "parakeet-v0.6.1-lib-linux-cpu-x64.tar.gz": (
        "https://github.com/mudler/parakeet.cpp/releases/download/v0.6.1/parakeet-v0.6.1-lib-linux-cpu-x64.tar.gz",
        "5a2b439e83d1b6ab66c55bcfc48751ee48e53727d5b3ea33a6bc0e8626f1397c",
    ),
    "parakeet-v0.6.1-lib-linux-cpu-arm64.tar.gz": (
        "https://github.com/mudler/parakeet.cpp/releases/download/v0.6.1/parakeet-v0.6.1-lib-linux-cpu-arm64.tar.gz",
        "d9e56bab07f87b6c05593a039f7cfe5b6033995da80f21721f167bfb41698c42",
    ),
    "parakeet-v0.6.1-lib-linux-cuda-x64.tar.gz": (
        "https://github.com/mudler/parakeet.cpp/releases/download/v0.6.1/parakeet-v0.6.1-lib-linux-cuda-x64.tar.gz",
        "f717526c49d56cd4691390ca4c8e521a27eec698a93f3897bf3817ceb744378a",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(directory: Path) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(directory)): {"sha256": _sha256(path), "size": path.stat().st_size}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def build(bundle: Bundle) -> Path:
    from huggingface_hub import snapshot_download

    directory = WORK / bundle.name
    directory.mkdir(parents=True, exist_ok=True)
    notice = [
        f"# {bundle.name}",
        "",
        bundle.description,
        "",
        "| Source | Commit | Licence |",
        "|---|---|---|",
    ]
    for source in bundle.sources:
        print(f"{bundle.name}: {source.repo}@{source.commit}", flush=True)
        target = directory / source.into if source.into else directory
        snapshot_download(
            source.repo,
            revision=source.commit,
            local_dir=target,
            allow_patterns=source.allow,
            ignore_patterns=source.ignore,
        )
        # huggingface_hub's bookkeeping, not part of the model.
        shutil.rmtree(target / ".cache", ignore_errors=True)
        notice.append(
            f"| https://huggingface.co/{source.repo} | `{source.commit}` | {source.licence} |"
        )
    others = bundle.extra(directory) if bundle.extra else []
    for other in others:
        notice.append(f"| {other['url']} | `{other['path']}` | {other['licence']} |")
    notes = [source.note for source in bundle.sources if source.note]
    notice += ["", *notes] if notes else []
    (directory / "NOTICE.md").write_text("\n".join(notice) + "\n")
    return directory


def push(bundle: Bundle, directory: Path) -> dict[str, object]:
    files = _files(directory)
    reference = f"{REGISTRY}/models/{bundle.name}:{bundle.tag}"
    print(f"{bundle.name}: pushing {len(files)} files to {reference}", flush=True)
    result = subprocess.run(
        [
            "oras",
            "push",
            reference,
            "--artifact-type",
            ARTIFACT_TYPE,
            "--annotation",
            f"org.opencontainers.image.source={SOURCE}",
            "--annotation",
            f"org.opencontainers.image.description={bundle.description}",
            "--format",
            "json",
            *[f"{path}:application/octet-stream" for path in files],
        ],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )
    digest = json.loads(result.stdout)["digest"]
    return {
        "ref": f"{REGISTRY}/models/{bundle.name}:{bundle.tag}@{digest}",
        "used_by": bundle.used_by,
        "sources": [
            {"repo": source.repo, "commit": source.commit, "licence": source.licence}
            for source in bundle.sources
        ],
        "files": files,
    }


def images() -> None:
    lock = _lock()
    for source, target in IMAGES.items():
        destination = f"{REGISTRY}/{target}"
        print(f"{source} -> {destination}", flush=True)
        subprocess.run(["oras", "cp", source, destination], check=True)
        digest = subprocess.run(
            ["oras", "resolve", destination], check=True, capture_output=True, text=True
        ).stdout.strip()
        lock["images"][source] = f"{destination}@{digest}"
    for target, parts in MULTIARCH.items():
        platforms = []
        for part in parts:
            index = json.loads(
                subprocess.run(
                    ["oras", "manifest", "fetch", f"{REGISTRY}/{part}"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            # The images' platform manifests, not their attestations.
            platforms += [
                f"{REGISTRY}/{part.split(':')[0]}@{entry['digest']}"
                for entry in index.get("manifests", [])
                if entry.get("platform", {}).get("os") == "linux"
            ]
        destination = f"{REGISTRY}/{target}"
        print(f"{' + '.join(parts)} -> {destination}", flush=True)
        subprocess.run(
            ["docker", "buildx", "imagetools", "create", "-t", destination, *platforms],
            check=True,
        )
        digest = subprocess.run(
            ["oras", "resolve", destination], check=True, capture_output=True, text=True
        ).stdout.strip()
        lock["images"][target] = f"{destination}@{digest}"
    _write(lock)


def vendor() -> None:
    lock = _lock()
    directory = WORK / "vendor"
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in VENDOR_FILES.items():
        path = directory / name
        if not path.exists():
            print(f"vendor: {url}", flush=True)
            with urllib.request.urlopen(url) as response, path.open("wb") as out:
                shutil.copyfileobj(response, out)
        if _sha256(path) != sha256:
            sys.exit(f"vendor: {name} does not match its sha256")
    (directory / "Dockerfile").write_text(
        "FROM scratch\n" + "".join(f"COPY {name} /{name}\n" for name in VENDOR_FILES)
    )
    reference = f"{REGISTRY}/vendor/{VENDOR_TAG}"
    # Both platforms (the same files): a build for either can COPY --from it.
    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64,linux/arm64",
            "--label",
            f"org.opencontainers.image.source={SOURCE}",
            "--provenance=false",
            "--push",
            "-t",
            reference,
            str(directory),
        ],
        check=True,
    )
    digest = subprocess.run(
        ["oras", "resolve", reference], check=True, capture_output=True, text=True
    ).stdout.strip()
    lock["vendor"] = {
        "ref": f"{reference}@{digest}",
        "files": {
            name: {"url": url, "sha256": sha256} for name, (url, sha256) in VENDOR_FILES.items()
        },
    }
    _write(lock)


def _lock() -> dict:
    if LOCK.exists():
        return json.loads(LOCK.read_text())
    return {"bundles": {}, "images": {}, "vendor": {}}


def _write(lock: dict) -> None:
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "push"):
        command = sub.add_parser(name)
        command.add_argument("bundles", nargs="*", choices=list(BUNDLES), default=[])
    sub.add_parser("images")
    sub.add_parser("vendor")
    arguments = parser.parse_args()
    if arguments.command == "images":
        images()
        return 0
    if arguments.command == "vendor":
        vendor()
        return 0
    lock = _lock()
    for name in arguments.bundles or list(BUNDLES):
        directory = build(BUNDLES[name])
        if arguments.command == "push":
            lock["bundles"][name] = push(BUNDLES[name], directory)
            _write(lock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
