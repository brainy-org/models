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
    uv run mirror.py ci RUN                take the libraries a run of native.yml built into .work/

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
import os
import re
import shlex
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
    # For a library built here: what it is built from and for, recorded in
    # the lock.
    built_from: NativeBuild | None = None
    platform: Platform | None = None
    # The package on ghcr.io, when it is not `name`: a library's bundles,
    # one per platform, are tags of one package.
    package: str = ""


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


@dataclass(frozen=True)
class NativeBuild:
    """A C/C++ library built from a commit with cmake, for a consumer that
    downloads a library rather than builds one (arvel). The source is the
    commit (and the patches kept in patches/); the library is built for one
    platform at a time (`Platform`), so a rebuild elsewhere gives other
    bytes and a new digest: a built file already in .work/ is reused, so
    pushing the bundle again keeps it."""

    repo: str
    commit: str
    licence: str
    # The C header the consumer declares its FFI from, with its sha256: the
    # bundle carries it, and the build stops if the commit's differs.
    header: str
    header_sha256: str
    # The macro that marks the header's exported functions: the smoke check
    # looks each one up in the built library.
    api: str
    target: str
    # The folder with the CMakeLists.txt, inside the repository.
    source: str = ""
    patches: tuple[str, ...] = ()
    # A function of the API that takes nothing and returns a static string,
    # called by the smoke check once the library is loaded.
    probe: str = ""


@dataclass(frozen=True)
class Platform:
    """Where a `NativeBuild` runs, and how it is built for it."""

    # The tag's suffix and the bundle's: `linux-vulkan-x64`.
    name: str
    # The host that builds it: platform.system(), platform.machine().
    host: tuple[str, str]
    # The library as built (under the build folder) and its name in the bundle.
    built: str
    library: str
    cmake_args: tuple[str, ...]
    # More patches, for this platform only.
    patches: tuple[str, ...] = ()
    # Files of the CUDA toolkit ($CUDA_PATH) shipped beside the library: the
    # runtime it links (NVIDIA's redistributables), found there by its rpath
    # ($ORIGIN) on Linux, by a load from the library's folder on Windows.
    redist: tuple[str, ...] = ()


# CUDA 12.9 (.github/workflows/native.yml installs it): Turing, Ampere, Ada
# and Blackwell (consumer) as SASS, and Ada's PTX, which the driver compiles
# for Hopper and newer. Pascal is left out (CUDA 13 drops it too).
CUDA_VERSION = "12.9.1"
CUDA_ARCHITECTURES = "75-real;80-real;86-real;89;120a-real"
CUDA_LICENCE = (
    "NVIDIA CUDA Toolkit EULA: cudart, cuBLAS and cuBLASLt are redistributable "
    "(Attachment A, https://docs.nvidia.com/cuda/eula/#attachment-a)"
)
# The CPU baseline off macOS: AVX2 with FMA and F16C on x64 (Haswell, Zen 1
# and newer), not ggml's per-ISA variants loaded at run time (one file is
# what a consumer pins); ARMv8.2-A with dotprod and fp16 on arm64, as on the
# Mac (Graviton 2, Ampere Altra, Cortex-A76 and newer).
X64 = ("-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_BMI2=ON")
ARM64 = ("-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16",)
# ggml's own thread pool, as on the Mac: no libgomp or vcomp140.dll to ship.
PORTABLE = ("-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF")
# Linux: libstdc++ linked in, and the static archives' symbols (ggml's)
# kept local, so a process that loads another ggml (libparakeet) does not
# bind this library's calls to it.
LINUX = ("-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL",)
# Windows: the C runtime linked in (/MT), no VC++ redistributable needed.
WINDOWS = ("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",)
CUDA = ("-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}")
LINUX_X64 = ("Linux", "x86_64")
LINUX_ARM64 = ("Linux", "aarch64")
WINDOWS_X64 = ("Windows", "AMD64")


def _native(
    spec: NativeBuild, platform_: Platform, package: str, name: str
) -> Callable[[Path], list[dict[str, str]]]:
    """The library of `spec` for `platform_`, built on such a host, with its
    header, patches and CUDA runtime beside it. A library already in the
    bundle's folder (built here before, or by CI and taken in with
    `mirror.py ci`) is kept as it is."""

    def fetch(directory: Path) -> list[dict[str, str]]:
        import platform

        patches = [*spec.patches, *platform_.patches]
        library = directory / platform_.library
        header = directory / Path(spec.header).name
        notes = [
            {
                "url": f"{spec.repo} at {spec.commit}",
                "path": platform_.library,
                "licence": spec.licence,
            }
        ]
        for patch in patches:
            shutil.copy(ROOT / patch, directory / Path(patch).name)
            notes.append(
                {
                    "url": f"{SOURCE}/blob/main/{patch}",
                    "path": Path(patch).name,
                    "licence": "applied to the source before the build",
                }
            )
        for file in platform_.redist:
            notes.append(
                {
                    "url": f"CUDA Toolkit {CUDA_VERSION}",
                    "path": Path(file).name,
                    "licence": CUDA_LICENCE,
                }
            )
        built = [library, header, *(directory / Path(f).name for f in platform_.redist)]
        if all(path.exists() for path in built):
            return notes
        if (platform.system(), platform.machine()) != platform_.host:
            sys.exit(f"{name}: built on {' '.join(platform_.host)} only")
        src = WORK / "src" / package
        # The files as committed (LF): the header's sha256 and the patches
        # are of those bytes, also on Windows.
        lf = ["git", "-c", "core.autocrlf=false"]
        if not (src / ".git").exists():
            subprocess.run([*lf, "clone", spec.repo, str(src)], check=True)
        git = [*lf, "-C", str(src)]
        subprocess.run([*git, "fetch", "--depth", "1", "origin", spec.commit], check=False)
        subprocess.run([*git, "checkout", "-q", "--force", spec.commit], check=True)
        subprocess.run(
            [*git, "submodule", "update", "--init", "--recursive", "--depth", "1"], check=True
        )
        if _sha256(src / spec.header) != spec.header_sha256:
            sys.exit(f"{name}: {spec.header} at {spec.commit} is not the one pinned")
        for patch in patches:
            path = str(ROOT / patch)
            applied = subprocess.run(
                [*git, "apply", "--reverse", "--check", path], check=False, capture_output=True
            )
            if applied.returncode != 0:
                subprocess.run([*git, "apply", path], check=True)
        source = src / spec.source
        build_dir = source / "build" / platform_.name
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build_dir),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                *platform_.cmake_args,
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build_dir), "--target", spec.target], check=True)
        # The real file, not the versioned symlinks that point at it.
        shutil.copy((build_dir / platform_.built).resolve(), library)
        shutil.copy(src / spec.header, header)
        if platform_.redist:
            cuda = Path(os.environ["CUDA_PATH"])
            for file in platform_.redist:
                shutil.copy((cuda / file).resolve(), directory / Path(file).name)
        record = {
            "built_with": shlex.join(platform_.cmake_args),
            "host": platform.platform(),
            **_toolchain(build_dir),
            **_smoke(spec, library),
        }
        _ci(record)
        (WORK / f"{name}.build.json").write_text(json.dumps(record, indent=2) + "\n")
        return notes

    return fetch


def _toolchain(build_dir: Path) -> dict[str, object]:
    """The compilers cmake used, with their versions, and the Vulkan SDK."""
    found = {}
    for language in ("CXX", "CUDA"):
        for path in build_dir.glob(f"CMakeFiles/*/CMake{language}Compiler.cmake"):
            text = path.read_text(errors="replace")
            values = dict(
                re.findall(
                    rf'^set\(CMAKE_{language}_COMPILER(_ID|_VERSION) "(.*)"\)', text, re.MULTILINE
                )
            )
            found[language.lower()] = f"{values.get('_ID', '?')} {values.get('_VERSION', '?')}"
    # Set by native.yml in the Vulkan jobs only.
    if os.environ.get("VULKAN_SDK"):
        found["vulkan_sdk"] = Path(os.environ["VULKAN_SDK"]).as_posix()
    return {"toolchain": found}


def _ci(record: dict[str, object]) -> None:
    """The CI run and runner image, when the build ran on GitHub Actions."""

    if os.environ.get("GITHUB_RUN_ID"):
        record["run"] = (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
        record["commit"] = os.environ.get("GITHUB_SHA", "")
        record["runner"] = f"{os.environ.get('ImageOS', '?')} {os.environ.get('ImageVersion', '?')}"


def _smoke(spec: NativeBuild, library: Path) -> dict[str, object]:
    """Load the library and resolve every function its header exports (and
    call the probe); where it cannot load (a GPU build without the driver's
    library), read the names from its symbol table instead. Stops the build
    if a function is missing, or, on Linux, if ggml's symbols are exported."""
    import ctypes

    names = re.findall(
        rf"^\s*{spec.api}\b[^;(]*?\b(\w+)\s*\(",
        (library.parent / Path(spec.header).name).read_text(),
        re.MULTILINE,
    )
    if not names:
        sys.exit(f"{library.name}: no {spec.api} function in the header")
    result: dict[str, object] = {"functions": len(names)}
    windows = sys.platform == "win32"
    for preload in filter(None, os.environ.get("MIRROR_SMOKE_PRELOAD", "").split(os.pathsep)):
        ctypes.CDLL(preload, mode=ctypes.RTLD_GLOBAL)
    if windows:
        for folder in filter(None, os.environ.get("MIRROR_SMOKE_DLL_DIRS", "").split(os.pathsep)):
            os.add_dll_directory(folder)
    try:
        loaded = ctypes.CDLL(str(library))
    except OSError as error:
        loaded = None
        result["load"] = f"failed: {error}"
    if loaded is not None:
        missing = [name for name in names if not hasattr(loaded, name)]
        result["load"] = "ok"
        if spec.probe and not missing:
            probe = getattr(loaded, spec.probe)
            probe.restype = ctypes.c_char_p
            result["probe"] = f"{spec.probe}() = {probe().decode()}"
        result["check"] = "loaded, every function resolved"
    else:
        if windows:
            out = subprocess.run(
                ["dumpbin", "/exports", str(library)], capture_output=True, text=True, check=True
            ).stdout
            exported = set(
                re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\w+)", out, re.MULTILINE)
            )
        else:
            out = subprocess.run(
                ["nm", "-D", "--defined-only", str(library)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            exported = {line.split()[-1] for line in out.splitlines() if line.strip()}
        missing = [name for name in names if name not in exported]
        result["check"] = "symbol table (the library cannot load here), every function exported"
    if missing:
        sys.exit(f"{library.name}: missing {', '.join(missing)}")
    if windows:
        out = subprocess.run(
            ["dumpbin", "/dependents", str(library)], capture_output=True, text=True, check=True
        ).stdout
        result["needs"] = sorted(
            set(re.findall(r"^\s+(\S+\.dll)\s*$", out, re.MULTILINE | re.IGNORECASE))
        )
    elif sys.platform.startswith("linux"):
        out = subprocess.run(
            ["nm", "-D", "--defined-only", str(library)], capture_output=True, text=True, check=True
        ).stdout
        leaked = [
            line.split()[-1] for line in out.splitlines() if re.search(r"\s(ggml|gguf)_", line)
        ]
        if leaked:
            sys.exit(f"{library.name} exports ggml's symbols: {', '.join(leaked[:5])}…")
        out = subprocess.run(
            ["readelf", "-d", str(library)], capture_output=True, text=True, check=True
        ).stdout
        result["needs"] = re.findall(r"\(NEEDED\).*\[(.+)\]", out)
        result["rpath"] = re.findall(r"\((?:RUNPATH|RPATH)\).*\[(.+)\]", out)
        out = subprocess.run(
            ["objdump", "-T", str(library)], capture_output=True, text=True, check=True
        ).stdout
        glibc = sorted(
            {tuple(map(int, v.split("."))) for v in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", out)}
        )
        result["glibc"] = ".".join(map(str, glibc[-1])) if glibc else ""
    print(f"{library.name}: {json.dumps(result)}", flush=True)
    return {"smoke": result}


# The libraries arvel loads with dlopen. arvel's FFI follows each header
# (src/speak/qwen.rs, src/audio/localvqe.rs): moving a commit means diffing
# the header. Neither header uses C's `long` (32 bits on Windows, 64 on
# Linux and macOS): the same declarations hold on every platform.
LIBQWEN = NativeBuild(
    repo="https://github.com/ServeurpersoCom/qwentts.cpp",
    commit="5916bf7c79dbb4e66fc059cdc507a79a23a28bb4",
    licence="MIT (qwentts.cpp, ggml)",
    header="src/qwen.h",
    header_sha256="08841d6299809754b2f6e4dbfce0bd1035ea351fcc51a0b437aeb39b48d79a97",
    api="QT_API",
    target="qwen",
    probe="qt_version",
)
# ggml linked in (BUILD_SHARED_LIBS=OFF): one library.
QWEN = ("-DQWEN_SHARED=ON", "-DBUILD_SHARED_LIBS=OFF")
LIBQWEN_PLATFORMS = (
    # Metal. Not -mcpu=native (ggml's default): the build machine's CPU
    # features would crash it on an older Mac. Every Apple silicon chip has
    # armv8.2-a with dotprod and fp16; the talker runs on the GPU anyway.
    Platform(
        "macos-arm64",
        ("Darwin", "arm64"),
        "libqwen.dylib",
        "libqwen.dylib",
        (
            "-DQWEN_SHARED=ON",
            "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_NATIVE=OFF",
            "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16",
        ),
    ),
    Platform(
        "linux-cpu-x64", LINUX_X64, "libqwen.so", "libqwen.so", (*QWEN, *PORTABLE, *X64, *LINUX)
    ),
    Platform(
        "linux-vulkan-x64",
        LINUX_X64,
        "libqwen.so",
        "libqwen.so",
        (*QWEN, *PORTABLE, *X64, *LINUX, "-DGGML_VULKAN=ON"),
    ),
    Platform(
        "linux-cuda-x64",
        LINUX_X64,
        "libqwen.so",
        "libqwen.so",
        (
            *QWEN,
            *PORTABLE,
            *X64,
            *LINUX,
            *CUDA,
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        ),
        redist=("lib/libcudart.so.12", "lib/libcublas.so.12", "lib/libcublasLt.so.12"),
    ),
    Platform(
        "linux-cpu-arm64",
        LINUX_ARM64,
        "libqwen.so",
        "libqwen.so",
        (*QWEN, *PORTABLE, *ARM64, *LINUX),
    ),
    Platform(
        "windows-cpu-x64", WINDOWS_X64, "qwen.dll", "qwen.dll", (*QWEN, *PORTABLE, *X64, *WINDOWS)
    ),
    Platform(
        "windows-vulkan-x64",
        WINDOWS_X64,
        "qwen.dll",
        "qwen.dll",
        (*QWEN, *PORTABLE, *X64, *WINDOWS, "-DGGML_VULKAN=ON"),
    ),
    Platform(
        "windows-cuda-x64",
        WINDOWS_X64,
        "qwen.dll",
        "qwen.dll",
        (*QWEN, *PORTABLE, *X64, *WINDOWS, *CUDA),
        redist=("bin/cudart64_12.dll", "bin/cublas64_12.dll", "bin/cublasLt64_12.dll"),
    ),
)
LIBLOCALVQE = NativeBuild(
    repo="https://github.com/localai-org/LocalVQE.git",
    commit="7f61cd05095d42e56a135b303f5bb4d761da22b9",
    licence="Apache-2.0 (LocalVQE), MIT (ggml)",
    header="ggml/localvqe_api.h",
    header_sha256="ef2871bde8cd8fe9aa8cd5c3b7d11cc72efb7bb1cc29f726522ace87ad490877",
    api="LOCALVQE_API",
    # Only the library is built (the bench targets download test files).
    # The configure step applies upstream's own ggml patch to its submodule.
    target="localvqe_shared",
    source="ggml",
    # Its own delay search (GCC-PHAT) off by default: arvel aligns the far
    # end itself, and the search once locked on a wrong delay for a whole
    # session (arvel's docs/decisions/2026-10-31-localvqe.md). The C API is
    # unchanged.
    patches=("patches/localvqe-no-prealign.patch",),
)
# The CPU only (parakeet and qwen have the GPU). LocalVQE's CMakeLists links
# ggml in statically, with GGML_NATIVE off, on arm64 itself; on x86 it
# builds ggml's per-ISA CPU variants as libraries loaded from beside it,
# which the patch turns into the arm64 path (one file).
LOCALVQE = ("-DLOCALVQE_BUILD_SHARED=ON",)
LOCALVQE_X64 = (*LOCALVQE, "-DLOCALVQE_X86_STATIC=ON", "-DGGML_OPENMP=OFF", *X64)
X86_STATIC = ("patches/localvqe-x86-static.patch",)
LIBLOCALVQE_PLATFORMS = (
    Platform(
        "macos-arm64",
        ("Darwin", "arm64"),
        "liblocalvqe.dylib",
        "liblocalvqe.dylib",
        (*LOCALVQE, "-DGGML_METAL=OFF"),
    ),
    Platform(
        "linux-cpu-x64",
        LINUX_X64,
        "liblocalvqe.so",
        "liblocalvqe.so",
        (*LOCALVQE_X64, *LINUX),
        patches=X86_STATIC,
    ),
    Platform(
        "linux-cpu-arm64",
        LINUX_ARM64,
        "liblocalvqe.so",
        "liblocalvqe.so",
        (*LOCALVQE, "-DGGML_OPENMP=OFF", *ARM64, *LINUX),
    ),
    Platform(
        "windows-cpu-x64",
        WINDOWS_X64,
        "localvqe.dll",
        "localvqe.dll",
        (*LOCALVQE_X64, *WINDOWS),
        patches=X86_STATIC,
    ),
)


def _native_bundles(
    name: str, spec: NativeBuild, platforms: tuple[Platform, ...], used_by: str, what: str
) -> list[Bundle]:
    """One bundle per platform but macOS (`libqwen` itself, defined on its
    own): `libqwen-<platform>`, each a tag of the same package."""
    bundles = []
    for platform_ in platforms[1:]:
        key = f"{name}-{platform_.name}"
        bundles.append(
            Bundle(
                name=key,
                package=name,
                tag=f"{spec.commit[:8]}-{platform_.name}",
                used_by=[used_by],
                description=f"{name}, {what}, built for {platform_.name}"
                f" ({platform_.library}, {Path(spec.header).name}"
                + "".join(f", {Path(p).name}" for p in (*spec.patches, *platform_.patches))
                + (", NVIDIA's CUDA runtime" if platform_.redist else "")
                + ")",
                extra=_native(spec, platform_, name, key),
                built_from=spec,
                platform=platform_,
            )
        )
    return bundles


# libparakeet v0.6.1, the builds brainy-worker's image and logbook load; the
# sha256 are the release's own (GitHub's asset digests, 2026-10-24).
PARAKEET_RELEASE = {
    "parakeet-v0.6.1-lib-linux-cpu-x64.tar.gz": "5a2b439e83d1b6ab66c55bcfc48751ee48e53727d5b3ea33a6bc0e8626f1397c",
    "parakeet-v0.6.1-lib-linux-cpu-arm64.tar.gz": "d9e56bab07f87b6c05593a039f7cfe5b6033995da80f21721f167bfb41698c42",
    "parakeet-v0.6.1-lib-linux-cuda-x64.tar.gz": "f717526c49d56cd4691390ca4c8e521a27eec698a93f3897bf3817ceb744378a",
    "parakeet-v0.6.1-lib-macos-metal-arm64.tar.gz": "1847118600359355c899619e263275d83b59f81dcc34fcca6ba4e6b9aac67db0",
    "parakeet-v0.6.1-lib-macos-cpu-x64.tar.gz": "4575814a5abb2d8d78b657750a3418cf2178765aa8f2f4ebfd692bf78fda9e7d",
    # Windows (2026-10-27): the Vulkan build loads the GPU driver's vulkan-1.dll.
    "parakeet-v0.6.1-lib-win-cpu-x64.zip": "95b41515efeab246094d1793a1c0a12d727bc1cc01e4c28fe37a16a994b26774",
    "parakeet-v0.6.1-lib-win-vulkan-x64.zip": "7ff5e20ecba4a2c78acec2917ee8ded9b416b52ba4e87e02bab43c02c2311722",
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
            used_by=["brainy-worker (recordings)", "logbook", "arvel (VAD, STT)"],
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
            used_by=["logbook (final pass: more than eight speakers)", "arvel (voice prints)"],
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
            used_by=["logbook (server, CLI and desktop app)", "arvel (macos-metal-arm64)"],
            description="libparakeet v0.6.1 release builds (parakeet.cpp's C API)",
            extra=_release(
                PARAKEET_RELEASE,
                "https://github.com/mudler/parakeet.cpp/releases/download/v0.6.1",
                "MIT (parakeet.cpp, ggml); the CUDA build carries NVIDIA's "
                "redistributable cudart and cuBLAS",
            ),
        ),
        Bundle(
            name="qwen3-tts",
            tag="b7ee2e8c",
            used_by=["arvel (TTS)"],
            description="Qwen3-TTS 12 Hz as qwentts.cpp loads it (GGUF): CustomVoice "
            "talkers 0.6B and 1.7B (Q8_0), speech tokenizer (F32)",
            sources=[
                # main on 2026-10-30: the GGUFs date from 968442208, this
                # commit only edits the README. The Qwen weights they are
                # converted from are Apache-2.0 too.
                HfSource(
                    "Serveurperso/Qwen3-TTS-GGUF",
                    "b7ee2e8c7459c3bea99da23e3d178125a7d1713c",
                    "Apache-2.0",
                    allow=[
                        "README.md",
                        "qwen-talker-0.6b-customvoice-Q8_0.gguf",
                        "qwen-talker-1.7b-customvoice-Q8_0.gguf",
                        "qwen-tokenizer-12hz-F32.gguf",
                    ],
                    note="Qwen3-TTS-12Hz-0.6B-CustomVoice, Qwen3-TTS-12Hz-1.7B-CustomVoice "
                    "and Qwen3-TTS-Tokenizer-12Hz by the Qwen team (Alibaba Cloud), "
                    "Apache-2.0, converted to GGUF by Serveurperso (qwentts.cpp).",
                ),
            ],
        ),
        Bundle(
            name="localvqe",
            tag="29ca3849",
            used_by=["arvel (--aec localvqe)"],
            description="LocalVQE v1.4-AEC, a neural acoustic echo canceller "
            "(200K parameters, F32 GGUF)",
            sources=[
                # main on 2026-10-31. Only the weights arvel loads: the
                # repository also holds older versions, PyTorch checkpoints
                # and the technical report. huggingface_hub checks the GGUF
                # against its LFS sha256 (b6e43138...3c731, arvel's pin).
                HfSource(
                    "LocalAI-io/LocalVQE",
                    "29ca38495cba9d6393a92a4dd890f28dd81f758d",
                    "Apache-2.0",
                    allow=["README.md", "localvqe-v1.4-aec-200K-f32.gguf"],
                    note="LocalVQE v1.4-AEC by Richard Palethorpe "
                    "(https://github.com/localai-org/LocalVQE), Apache-2.0, a streaming "
                    "derivative of DeepVQE (Indenbom et al., Interspeech 2023).",
                ),
            ],
        ),
        Bundle(
            name="libqwen",
            tag="5916bf7c-macos-arm64",
            used_by=["arvel (TTS)"],
            description="libqwen, qwentts.cpp's C API, built for macOS arm64 with Metal "
            "(the dylib and qwen.h)",
            extra=_native(LIBQWEN, LIBQWEN_PLATFORMS[0], "libqwen", "libqwen"),
            built_from=LIBQWEN,
            platform=LIBQWEN_PLATFORMS[0],
        ),
        Bundle(
            name="liblocalvqe",
            tag="7f61cd05-macos-arm64",
            used_by=["arvel (--aec localvqe)"],
            description="liblocalvqe, LocalVQE's C API, built for macOS arm64 on the CPU "
            "with arvel's patch (the dylib, localvqe_api.h and the patch)",
            extra=_native(LIBLOCALVQE, LIBLOCALVQE_PLATFORMS[0], "liblocalvqe", "liblocalvqe"),
            built_from=LIBLOCALVQE,
            platform=LIBLOCALVQE_PLATFORMS[0],
        ),
        *_native_bundles(
            "libqwen", LIBQWEN, LIBQWEN_PLATFORMS, "arvel (TTS)", "qwentts.cpp's C API"
        ),
        *_native_bundles(
            "liblocalvqe",
            LIBLOCALVQE,
            LIBLOCALVQE_PLATFORMS,
            "arvel (--aec localvqe)",
            "LocalVQE's C API on the CPU with arvel's patch",
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
        from huggingface_hub import snapshot_download

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
    reference = f"{REGISTRY}/models/{bundle.package or bundle.name}:{bundle.tag}"
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
    built = WORK / f"{bundle.name}.build.json"
    return {
        "ref": f"{reference}@{digest}",
        "used_by": bundle.used_by,
        "sources": [
            {"repo": source.repo, "commit": source.commit, "licence": source.licence}
            for source in bundle.sources
        ]
        + (
            [
                {
                    "repo": bundle.built_from.repo,
                    "commit": bundle.built_from.commit,
                    "licence": bundle.built_from.licence,
                    "built_with": shlex.join(bundle.platform.cmake_args),
                    **(
                        {"patches": [*bundle.built_from.patches, *bundle.platform.patches]}
                        if bundle.built_from.patches or bundle.platform.patches
                        else {}
                    ),
                    # Where and how it was built (the CI run, the runner,
                    # the toolchain, the smoke check), when it was recorded.
                    **({"built_on": json.loads(built.read_text())} if built.exists() else {}),
                }
            ]
            if bundle.built_from and bundle.platform
            else []
        ),
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


def ci(run: str) -> None:
    """Take the libraries a run of .github/workflows/native.yml built into
    .work/, each with the record of its build, for `push`. A bundle already
    in .work/ is kept: a build is not replaced by another one's bytes."""
    downloaded = WORK / "ci" / run
    if not downloaded.exists():
        subprocess.run(
            ["gh", "run", "download", run, "-R", "brainy-org/models", "-D", str(downloaded)],
            check=True,
        )
    for artifact in sorted(downloaded.iterdir()):
        name = artifact.name
        if name not in BUNDLES or not BUNDLES[name].platform:
            sys.exit(f"ci: {name} is not a library built here")
        if (WORK / name).exists():
            print(f"{name}: already in .work/, kept", flush=True)
            continue
        shutil.copytree(artifact / name, WORK / name)
        shutil.copy(artifact / f"{name}.build.json", WORK / f"{name}.build.json")
        print(f"{name}: taken from run {run}", flush=True)


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
    sub.add_parser("ci").add_argument("run", help="the id of a run of native.yml")
    arguments = parser.parse_args()
    if arguments.command == "ci":
        ci(arguments.run)
        return 0
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
