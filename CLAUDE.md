# models

The mirror of third-party model weights, images and build inputs on
ghcr.io/brainy-org. `README.md` says what is mirrored, how and who uses it;
`mirror.py` is the mirror's code; `models.lock.json` is the record it writes.
`finetune/` is apart: a recipe that tunes parakeet-tdt-0.6b-v3 for Italian
heard across a room, run on Kaggle (`finetune/README.md`); what it makes is
mirrored only once it wins on the owner's recordings, as any other bundle,
and its pins follow the same rule below. English only, in code and
documentation.

## Commands

```bash
uv run mirror.py build <bundle>    # fetch and check into .work/, no push
uv run mirror.py push <bundle>     # needs `oras login ghcr.io` (write:packages)
uv run mirror.py images | vendor
uv run mirror.py ci <run id>       # libraries built by native.yml into .work/
uvx ruff check .                   # mirror.py and finetune/ (ruff.toml)
```

## Invariants

- **A source is a commit or a file with its sha256, never a branch, tag or
  pull-request ref.** Those move: what is mirrored must be what was
  reviewed.
- **Consumers pin digests, and a digest is never deleted.** Pushing a
  changed bundle under the same tag leaves the old digest in place; deleting
  a package version breaks every installation that pinned it.
- **A licence travels with the weights**: every bundle has a `NOTICE.md`,
  and a model whose licence forbids redistribution is not mirrored.
- **Changing a pin is a change in every consumer** (README, "Consumers"):
  brainy-worker's `models.json`, the charts' `tei.*`, the compose files,
  logbook's `crates/engine/src/mirror.rs`, arvel's
  `src/setup/mirror/pins.rs`. One
  commit per repository.
- **The `docling` bundle follows brainy-worker's `uv.lock`**: the docling
  and rapidocr versions in `mirror.py`'s header must match it, or the bundle
  lacks files that docling asks for (and the worker, offline, fails).
