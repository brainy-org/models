# brainy-org/models

The mirror of what the brainy-org services download from third parties:
model weights, the images that serve them, and the build inputs. Everything
lives on ghcr.io/brainy-org as packages (private, but for logbook's:
"Access"), pinned by digest, so a
model that its author deletes, moves or rewrites changes nothing for us.

| What | Where | Used by |
|---|---|---|
| `docling` bundle: layout (heron), TableFormer ACCURATE, CodeFormulaV2, RapidOCR (PP-OCRv6 small and the v5/v4 recogniser of each other script) | `ghcr.io/brainy-org/models/docling` | brainy-worker (docling's `artifacts_path`) |
| `parakeet` bundle: ASR, VAD, diarization (GGUF) | `ghcr.io/brainy-org/models/parakeet` | brainy-worker (recordings), logbook, arvel |
| `qwen3-embedding-0.6b` bundle: Qwen3-Embedding-0.6B and its ONNX export | `ghcr.io/brainy-org/models/qwen3-embedding-0.6b` | brainy's and Galley's TEI |
| `parakeet-voice` bundle: WeSpeaker ResNet34-LM speaker encoder (GGUF), the "voice" component of mudler's `parakeet-bundle-small` written back as its standalone file, byte for byte | `ghcr.io/brainy-org/models/parakeet-voice` | logbook (final pass speakers) |
| `libparakeet` bundle: parakeet.cpp's release builds | `ghcr.io/brainy-org/models/libparakeet` | logbook, arvel |
| `qwen3-tts` bundle: Qwen3-TTS 12 Hz for qwentts.cpp (GGUF): CustomVoice talkers 0.6B and 1.7B (Q8_0), speech tokenizer (F32) | `ghcr.io/brainy-org/models/qwen3-tts` | arvel (TTS) |
| `localvqe` bundle: LocalVQE v1.4-AEC, a neural acoustic echo canceller (F32 GGUF, 200K parameters) | `ghcr.io/brainy-org/models/localvqe` | arvel (`--aec localvqe`) |
| `libqwen` bundles: qwentts.cpp's C API built here, one tag per platform: macOS arm64 with Metal; Linux x64 on the CPU, Vulkan or CUDA; Linux arm64 on the CPU; Windows x64 on the CPU, Vulkan or CUDA; each with its `qwen.h` (the CUDA ones with NVIDIA's runtime) | `ghcr.io/brainy-org/models/libqwen` | arvel (TTS) |
| `liblocalvqe` bundles: LocalVQE's C API built here on the CPU with `patches/localvqe-no-prealign.patch`, one tag per platform (macOS arm64, Linux x64 and arm64, Windows x64), each with its `localvqe_api.h` and the patches | `ghcr.io/brainy-org/models/liblocalvqe` | arvel (`--aec localvqe`) |
| `face-landmarker` bundle: MediaPipe Face Landmarker (float16, version 1), the `.task` file Google publishes at a versioned URL (face detector, face mesh, blendshapes; Apache-2.0) | `ghcr.io/brainy-org/models/face-landmarker` | arvel (`--gaze`, served to the satellite's page) |
| `arvel-sandbox-guest` bundles: arvel-sandbox's guest, one tag per architecture (`<commit>-linux-arm64`, `-amd64`): the kernel (Alpine's linux-virt), the initramfs, the sandbox's and the toolbox's Debian root disks with Playwright and Debian's chromium (zstd), the same roots as tars for a pod's image, the package lists, `guest.json`. Built by arvel-sandbox, not here | `ghcr.io/brainy-org/models/arvel-sandbox-guest` | arvel-sandbox (`sandbox-vm`'s guest) |
| Images: Text Embeddings Inference (amd64, arm64), oras | `ghcr.io/brainy-org/mirror/*` | the Helm charts and compose files of brainy and Galley |
| Build inputs: ffmpeg's source, libparakeet | `ghcr.io/brainy-org/vendor/brainy-worker-build` | brainy-worker's Dockerfile (`VENDOR_IMAGE`) |

`models.lock.json` is the record: each bundle's reference with its digest,
its sources (repository, commit, licence) and every file with its sha256
and size; each image's mirrored digest; the build inputs with their
upstream URL and sha256. Every bundle carries a `NOTICE.md` with the source
and licence of each file (CC-BY-4.0 asks for the attribution).

## How a bundle is made

A bundle is an OCI artifact (`application/vnd.brainy-org.model-bundle.v1`):
one layer per file, titled with its path. `mirror.py` builds it from
**commits**, never a branch or a tag (Hugging Face's `main`, a pull
request's `refs/pr/N` and tags all move), checks every file it can against
an upstream sha256 (RapidOCR's `default_models.yaml`, GitHub's release
digests), writes the `NOTICE.md`, and pushes it with oras. The `docling`
bundle is laid out as docling's `artifacts_path` expects, by the same
docling and rapidocr versions as brainy-worker's `uv.lock` (the script's
inline dependencies): their resolvers decide which files a language needs.

`libqwen` and `liblocalvqe` are the exception to "fetch, don't build": no
upstream publishes the libraries arvel loads, so `mirror.py` builds them
(`NativeBuild`: a commit, the header's sha256 checked first, patches from
`patches/`; a `Platform` for each target: its cmake flags and host). Each
platform is its own bundle and tag of the package (`libqwen` is macOS,
`libqwen-linux-vulkan-x64` is the tag `5916bf7c-linux-vulkan-x64`). The
macOS ones build on a Mac with Apple silicon, cmake and ninja; the Linux
and Windows ones on GitHub Actions (`.github/workflows/native.yml`: Ubuntu
22.04 for glibc 2.34 and newer, its arm64 runner, Windows Server 2022 with
MSVC), which uploads each as an artifact and pushes nothing:

```bash
gh workflow run native.yml -R brainy-org/models                 # every one, or -f bundles="libqwen-linux-cpu-x64 ..."
gh run watch <run id> -R brainy-org/models
uv run mirror.py ci <run id>      # the artifacts into .work/, each with the record of its build
uv run mirror.py push libqwen-linux-cpu-x64 ...
```

Every build is one self-contained library: ggml linked in statically
(GGML_NATIVE off, ggml's own thread pool rather than OpenMP), libstdc++
linked in and ggml's symbols kept local on Linux, the C runtime linked in
(/MT) on Windows. It needs only the C library, and `libvulkan.so.1` /
`vulkan-1.dll` (the GPU driver's loader) for Vulkan; the CUDA builds need
the driver's `libcuda.so.1` / `nvcuda.dll` and carry NVIDIA's runtime
beside the library (cudart, cuBLAS, cuBLASLt, CUDA 12.9.1,
redistributable under NVIDIA's EULA): an rpath of `$ORIGIN` finds it on
Linux, a load with the library's folder in the search path on Windows. The
CPU baseline is AVX2 with FMA and F16C on x64 and ARMv8.2-A with dotprod
and fp16 on arm64; the CUDA builds carry SASS for Turing, Ampere, Ada and
consumer Blackwell (sm 75, 80, 86, 89, 120a) and Ada's PTX for Hopper and
newer. LocalVQE's own build makes ggml's per-ISA CPU variants on x86 as
libraries loaded from beside it; `patches/localvqe-x86-static.patch` makes
it link one in, as it does on arm64. (MSVC ignores the
`-ffp-contract=off` it asks for its adaptive filter, with a warning: MSVC
does not contract floating point by default.) Before it is uploaded each library is
loaded and every function of its header resolved (`qt_version()` called);
a CUDA build on Windows, which cannot load without the driver, is checked
on its export table instead (the Linux one loads with the toolkit's stub
of the driver).

`arvel-sandbox-guest` is the other exception: arvel-sandbox builds it (its
`guest/build.sh`, `guest/disks.sh` and `guest/bundle.sh`; on a Mac in a
builder VM, on Linux in a chroot), from pinned inputs (Alpine's and static
curl's files by sha256, Debian's layer by digest, Playwright by version; the
Debian packages, chromium among them, are those of the day, upgraded) and records them in the
bundle's `guest.json`. `mirror.py guest DIR` takes the folder `bundle.sh`
made, writes its `NOTICE.md` from `guest.json`, pushes it as
`arvel-sandbox-guest:<commit>-linux-<arch>` (`-dirty` when the tree had
changes) and records it as `arvel-sandbox-guest-<arch>`. It needs nothing but
oras and Python's standard library (`python3 mirror.py guest DIR` works
without uv). arvel-sandbox's `.github/workflows/guest.yml` does this on
GitHub's Linux runners for both architectures and opens the lock's pull
request here; by hand, on a Mac:

```bash
(cd ../arvel-sandbox && make guest && ROOTFS_TAR=1 make disks && make bundle)
python3 mirror.py guest ../arvel-sandbox/guest/out/bundle-arm64
```

The build is not reproducible bit for bit: a build already in `.work/` is
reused, and the lock records what it was built from (`built_on`: the CI
run, its commit, the runner image, the compilers, the smoke check, the
libraries it needs).

```bash
gh auth refresh -s write:packages
gh auth token | oras login ghcr.io -u "$(gh api user -q .login)" --password-stdin
gh auth token | docker login ghcr.io -u "$(gh api user -q .login)" --password-stdin

uv run mirror.py build docling        # fetch into .work/ and check, without pushing
uv run mirror.py push docling         # build, push, update models.lock.json
uv run mirror.py push                 # every bundle
uv run mirror.py images               # copy the third-party images
uv run mirror.py vendor               # push the build inputs image
```

A push of unchanged files gives the same digest. A changed one gives a new
digest under the same tag: the consumers keep the old one until they are
moved, and old digests are never deleted.

## Consumers

Each consumer pins what it uses, copied from `models.lock.json`:

| Consumer | Where the pin lives | How it pulls |
|---|---|---|
| brainy-worker | `src/brainy_worker/models.json` (ref and files of `docling`, `parakeet`); `VENDOR_IMAGE` in the Dockerfile | `brainy-worker models` (its own OCI client), in an init container in the charts |
| brainy, Galley (Helm) | `tei.bundle`, `tei.revision`, `tei.image`, `tei.puller.image` in `values.yaml` | an oras init container, with the `ghcr-pull` secret |
| brainy, Galley (compose) | the `tei-model` service, `*_TEI_IMAGE` | the same, with `GHCR_USER`/`GHCR_TOKEN` from `.env` |
| logbook | the files' sha256 in `crates/engine/src/mirror.rs` (`parakeet`, `parakeet-voice`, `libparakeet`) | logbook-server itself, by blob digest, on first start |
| arvel-sandbox | the manifest digest per architecture in `cmd/sandbox-vm/guestpin.go` (`arvel-sandbox-guest`) | the `sandbox-vm` binary itself, at a start, into `~/.cache/brainy-org/arvel-sandbox-guest-<digest>/` |
| arvel | manifest digests and files' sha256 in `src/setup/mirror/pins.rs` (`parakeet`, `libparakeet`, `qwen3-tts`, `localvqe`, `libqwen`, `liblocalvqe`, `face-landmarker`) | the `arvel` binary itself, by blob digest, on first start, into `~/.cache/brainy-org` (logbook's folders for what both load) |
| the `ae` VM's TEI | brainy's `helm/README.md` | `oras pull` |

## Adding or changing a model

1. Check its licence allows redistribution, and add it to brainy-worker's
   licences record (`docs/decisions/2026-10-10-licences-of-code-and-weights.md`)
   if the worker loads it.
2. Add or change the bundle in `mirror.py`, by commit; `uv run mirror.py push <name>`.
3. Move the consumers' pins (table above) in the same change, one commit
   per repository. A different embedding model or commit re-embeds every
   page and document once.

Upgrading docling in brainy-worker is step 2 for `docling`: bump the
versions in `mirror.py`'s header with `uv.lock`, push, copy the entry into
`models.json`, and run the worker's tests offline (its README, "Models").

## Fine-tuning

`finetune/` holds a recipe, not a bundle: it tunes `parakeet-tdt-0.6b-v3` on
public Italian speech put into simulated rooms (real and simulated room
impulse responses, noise, a low level), on Kaggle's free GPUs, and converts
the result to the q8_0 GGUF libparakeet loads, checked against the
`parakeet` bundle's. Its sources are pinned like the mirror's (commits and
sha256). A tuned model is mirrored here, as a new bundle with its
`NOTICE.md` (CC-BY-4.0 of the base model), only once it beats the current
one on the owner's own recordings. How to run it: `finetune/README.md`.

## Access

`models/parakeet`, `models/parakeet-voice` and `models/libparakeet` are
public since 2026-10-26: logbook's CLI fetches them on every user's
computer, with no token. Their licences allow it (each bundle's
`NOTICE.md`); a public package cannot be made private again, so a bundle
is made public only once its licences have been read for redistribution.
`models/qwen3-tts` is public since 2026-10-30: the arvel binary pulls it
with no token, and its licence (Apache-2.0, the GGUFs and the Qwen weights
they come from) allows it. `models/localvqe` (Apache-2.0, which allows it)
is public since 2026-10-31, for arvel too. `models/libqwen` (MIT) and
`models/liblocalvqe` (Apache-2.0 and MIT) are public, for the arvel binary
too; a tag pushed to a public package is public, the Linux and Windows
ones included (the CUDA ones carry NVIDIA's runtime, which its EULA lets
an application redistribute). `models/face-landmarker` (Apache-2.0, by
Google's model cards) is public since 2026-11-03, for arvel's `--gaze`. `models/arvel-sandbox-guest`
was public, and its versions built before 2026-11-22's fix are being deleted (the one exception
to "a digest is never deleted"): their roots carried Playwright's `chromium`, which is Google
Chrome for Testing with the Widevine CDM (`libwidevinecdm.so`), whose licence forbids
distribution. The roots now carry Debian's `chromium` package (BSD-3-Clause and others) as
Playwright's browser, and arvel-sandbox's build fails if a Google Chrome, Widevine or
Playwright-downloaded browser file is in a root (`guest/make-root`, `guest/bundle.sh`). Only
redistributable open-source software goes in it: the kernel and busybox (GPL-2.0, Alpine's),
Debian's packages, Playwright (Apache-2.0), arvel-sandbox's own code; each bundle's `NOTICE.md`
lists the sources and licences and carries the written offer of the GPL's corresponding source
(GPL-2.0 §3(b)). Making a new version public is the owner's decision once its `NOTICE.md` has
been read; until then `sandbox-vm` pulls it with a token (`SANDBOX_GHCR_TOKEN`). Its package grants arvel-sandbox's Actions write access (its
`guest.yml` pushes with `GITHUB_TOKEN`). The others are private, readable with the organization's pull secret
(`ghcr-pull` in the clusters) or a token with `read:packages`. GitHub
Actions read them with `GITHUB_TOKEN` only where a package grants the
repository access (package settings, "Manage Actions access"): brainy-worker
needs `models/docling`, `models/parakeet` and `vendor/brainy-worker-build`.

## Licence

The code here (`mirror.py`, `finetune/`, `patches/`, the workflows) is under the Apache License 2.0
(`LICENSE`). The bundles are not: each carries its own sources and licences in its `NOTICE.md`.
