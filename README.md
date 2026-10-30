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
| `qwen3-tts` bundle: Qwen3-TTS 12 Hz for qwentts.cpp (GGUF): CustomVoice talkers 0.6B and 1.7B (Q8_0), speech tokenizer (F32). Defined in `mirror.py`, not pushed yet: no entry in `models.lock.json` until it is | `ghcr.io/brainy-org/models/qwen3-tts` | arvel (TTS) |
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
| arvel | the files' sha256 in `scripts/fetch.sh` (`parakeet`, `libparakeet`, `qwen3-tts`) | `scripts/fetch.sh`, by blob digest with curl, into logbook's cache folders |
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

## Access

`models/parakeet`, `models/parakeet-voice` and `models/libparakeet` are
public since 2026-10-26: logbook's CLI fetches them on every user's
computer, with no token. Their licences allow it (each bundle's
`NOTICE.md`); a public package cannot be made private again, so a bundle
is made public only once its licences have been read for redistribution.
`models/qwen3-tts` is public since 2026-10-30: arvel's
`scripts/fetch.sh` pulls it with no token, and its licence (Apache-2.0,
the GGUFs and the Qwen weights they come from) allows it. The others are private, readable with the organization's pull secret
(`ghcr-pull` in the clusters) or a token with `read:packages`. GitHub
Actions read them with `GITHUB_TOKEN` only where a package grants the
repository access (package settings, "Manage Actions access"): brainy-worker
needs `models/docling`, `models/parakeet` and `vendor/brainy-worker-build`.
