"""What prepare.py, train.py and evaluate.py share: the pins, the folders, the
manifests, the text normalisation for WER and the simulated room.

Every source is a commit or a file with its sha256, as in mirror.py: a
Hugging Face revision is a commit, never `main`; a file from elsewhere is
checked before it is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The model this recipe tunes: the same weights the mirror's `parakeet` bundle
# was converted from (mudler/parakeet-cpp-gguf's tdt-0.6b-v3), CC-BY-4.0.
BASE_REPO = "nvidia/parakeet-tdt-0.6b-v3"
BASE_COMMIT = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
BASE_FILE = "parakeet-tdt-0.6b-v3.nemo"
BASE_SHA256 = "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d"

# Hugging Face datasets, read from their parquet files at a commit.
VOXPOPULI = ("facebook/voxpopuli", "42f01879c780b4a2e90ec0b4f616c2ece526e4f1")  # CC0-1.0
FLEURS = ("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd")  # CC-BY-4.0
MLS = ("facebook/multilingual_librispeech", "2e83e61823b4c47dcbcb1980bb88601274127609")  # CC-BY-4.0

# OpenSLR SLR28, "Room Impulse Response and Noise Database" (Apache-2.0 on
# OpenSLR; the point-source noises are public domain Freesound recordings).
# OpenSLR publishes no checksum: this sha256 is the one two Hugging Face
# copies of the file record (dalietng/rirs_noises, EaseZh/rirs_noises), of the
# same size as openslr.org serves; prepare.py stops if the download differs.
RIRS_URLS = (
    "https://www.openslr.org/resources/28/rirs_noises.zip",
    "https://openslr.elda.org/resources/28/rirs_noises.zip",
)
RIRS_SHA256 = "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7"
RIRS_SIZE = 1311166223

SAMPLE_RATE = 16000


def work_dir(name: str) -> Path:
    """Kaggle's persisted output folder when there is one, else .work/ in this
    repository (ignored by git)."""
    base = Path("/kaggle/working")
    if not base.is_dir():
        base = ROOT.parent / ".work" / "finetune"
    return base / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_base(check: bool = True) -> Path:
    """The pinned .nemo, from the Hugging Face cache or downloaded into it."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(BASE_REPO, BASE_FILE, revision=BASE_COMMIT))
    if check and sha256(path) != BASE_SHA256:
        raise SystemExit(f"{path}: sha256 differs from the pinned {BASE_SHA256}")
    return path


# Manifests: NeMo's JSON lines. Paths are relative to the manifest's folder,
# which NeMo resolves (manifest.get_full_path), so a folder made in one Kaggle
# session still works when attached read-only to another.


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def absolute(manifest: Path, row: dict) -> str:
    audio = Path(row["audio_filepath"])
    return str(audio if audio.is_absolute() else manifest.parent / audio)


def clean_text(text: str) -> str:
    """Training targets: the source's casing and punctuation, since v3 writes
    both; only typographic quotes are folded and spaces collapsed."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_wer(text: str) -> str:
    """Lower case, no punctuation (an apostrophe splits: "l'uomo" is two words),
    one space: the same for reference and hypothesis, for every model."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)
    return " ".join(text.split())


def simulate_room(speech, rir, noise, rng, snr_db: tuple[float, float],
                  gain_db: tuple[float, float]):
    """One far-field version of `speech`, the way train.py's NeMo augmentors
    make one (ImpulsePerturbation, then NoisePerturbation, then
    GainPerturbation): convolved with the RIR and trimmed to its length, peak
    normalised, noise added at an SNR measured on RMS, then a gain. Arrays are
    float32 at 16 kHz; `rng` is a numpy Generator, so a seed fixes the result."""
    import numpy as np
    from scipy.signal import fftconvolve

    out = fftconvolve(speech, rir, mode="full")[: len(speech)]
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak
    if len(noise) < len(out):
        noise = np.tile(noise, int(np.ceil(len(out) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(out) + 1))
    noise = noise[start : start + len(out)]
    rms_s = np.sqrt(np.mean(out**2)) + 1e-12
    rms_n = np.sqrt(np.mean(noise**2)) + 1e-12
    snr = rng.uniform(*snr_db)
    out = out + noise * (rms_s / (rms_n * 10 ** (snr / 20)))
    out = out * 10 ** (rng.uniform(*gain_db) / 20)
    peak = np.max(np.abs(out))
    if peak > 0.99:  # written as 16-bit PCM: scale rather than clip
        out = out * (0.99 / peak)
    return out.astype(np.float32)
