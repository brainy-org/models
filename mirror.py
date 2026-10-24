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
