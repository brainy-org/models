# models

The mirror of third-party model weights, images and build inputs on
ghcr.io/brainy-org. `README.md` says what is mirrored, how and who uses it;
`mirror.py` is the only code; `models.lock.json` is the record it writes.
English only, in code and documentation.

## Commands

```bash
uv run mirror.py build <bundle>    # fetch and check into .work/, no push
uv run mirror.py push <bundle>     # needs `oras login ghcr.io` (write:packages)
uv run mirror.py images | vendor
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
  logbook's `scripts/server.sh`. One commit per repository.
- **The `docling` bundle follows brainy-worker's `uv.lock`**: the docling
  and rapidocr versions in `mirror.py`'s header must match it, or the bundle
  lacks files that docling asks for (and the worker, offline, fails).
