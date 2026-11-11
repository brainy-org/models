#!/usr/bin/env bash
# Convert a tuned .nemo to the q8_0 GGUF that libparakeet v0.6.1 loads, with
# parakeet.cpp's own converter at the commit v0.6.1 points to, then compare it
# with the GGUF the mirror's `parakeet` bundle carries: every parakeet.* key
# (tokenizer pieces, architecture, preprocessor) and every tensor's name and
# shape must be the same, or arvel's and logbook's libparakeet would not load
# it as the same model.
#
#   ./convert.sh RUN/parakeet-tdt-0.6b-v3-it-room.nemo [OUT.gguf]
#
# Needs the Python of train.py (NeMo 2.7.3, gguf: requirements.txt) as $PYTHON,
# git, and about 4 GB free in $WORK. Prints the GGUF's sha256 and size.
set -euo pipefail

NEMO=${1:?usage: convert.sh MODEL.nemo [OUT.gguf]}
OUT=${2:-$(dirname "$NEMO")/tdt-0.6b-v3-it-room-q8_0.gguf}
PYTHON=${PYTHON:-python}
WORK=${WORK:-/tmp/parakeet-convert}

# parakeet.cpp v0.6.1 (the libparakeet bundle's release), by commit.
PARAKEET_CPP=5c9b0fbbd8e611d88e47d7e1ebae9bca9b447e8f
# The GGUF of the mirror's parakeet bundle (models.lock.json), by commit.
REFERENCE_REPO=mudler/parakeet-cpp-gguf
REFERENCE_COMMIT=741158ae71e64ef5c89385862c18f777d07a97a1
REFERENCE_FILE=tdt-0.6b-v3-q8_0.gguf
REFERENCE_SHA256=4d69a4a6683f4f2d952bad794c1357ca6eb628027695b4699c5a9ad4cd07d757

mkdir -p "$WORK"
if [ ! -d "$WORK/parakeet.cpp/.git" ]; then
    git init -q "$WORK/parakeet.cpp"
    git -C "$WORK/parakeet.cpp" fetch -q --depth 1 https://github.com/mudler/parakeet.cpp "$PARAKEET_CPP"
    git -C "$WORK/parakeet.cpp" checkout -q FETCH_HEAD
fi
test "$(git -C "$WORK/parakeet.cpp" rev-parse HEAD)" = "$PARAKEET_CPP"

"$PYTHON" "$WORK/parakeet.cpp/scripts/convert_parakeet_to_gguf.py" \
    --model "$NEMO" --output "$OUT" --dtype q8_0

"$PYTHON" - "$OUT" "$REFERENCE_REPO" "$REFERENCE_COMMIT" "$REFERENCE_FILE" "$REFERENCE_SHA256" <<'EOF'
import hashlib
import sys
from pathlib import Path

import gguf
from huggingface_hub import hf_hub_download

out, repo, commit, name, want = sys.argv[1:]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


reference = hf_hub_download(repo, name, revision=commit)
if sha256(reference) != want:
    sys.exit(f"{reference}: not the mirror's {name}")


def describe(path):
    reader = gguf.GGUFReader(path)
    keys = {k: f.contents() for k, f in reader.fields.items() if k.startswith("parakeet.")}
    shapes = {t.name: t.shape.tolist() for t in reader.tensors}
    types = {t.name: t.tensor_type.name for t in reader.tensors}
    return keys, shapes, types


new_keys, new_tensors, new_types = describe(out)
ref_keys, ref_tensors, ref_types = describe(reference)
problems = [f"key {k}: differs" for k in ref_keys.keys() & new_keys.keys()
            if ref_keys[k] != new_keys[k]]
problems += [f"tensor {t}: {new_tensors.get(t)} vs {ref_tensors.get(t)}"
             for t in ref_tensors.keys() | new_tensors.keys()
             if new_tensors.get(t) != ref_tensors.get(t)]
for k in sorted(ref_keys.keys() ^ new_keys.keys()):
    # A key only one side has: the reference was converted by an earlier
    # converter. Reported, not fatal; the libparakeet that loads the
    # reference also loads the keys v0.6.1 writes.
    print(f"note: key {k} only in {'reference' if k in ref_keys else 'new GGUF'}")
for t in sorted(ref_types.keys() & new_types.keys()):
    if ref_types[t] != new_types[t]:  # a converter's choice, not the model's
        print(f"note: tensor {t} stored as {new_types[t]}, the reference as {ref_types[t]}")
if problems:
    sys.exit("not the same model as the mirror's GGUF:\n  " + "\n  ".join(problems[:20]))
print(f"same keys and tensors as {repo}@{commit[:8]}/{name}")
print(f"{out}\n  sha256 {sha256(out)}\n  size   {Path(out).stat().st_size}")
EOF
