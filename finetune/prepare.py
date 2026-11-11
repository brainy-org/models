"""Download the Italian speech, the room impulse responses and the noises, and
write them as 16 kHz mono WAVs with NeMo manifests.

    python prepare.py [--out DIR] [--voxpopuli-hours 60] [--mls-hours 0]

Needs huggingface_hub, pyarrow, soundfile, scipy and numpy (all in Kaggle's
image). No GPU: run it in a CPU session; its output folder becomes the input
of train.py and evaluate.py (README.md).

What it writes into DIR:
  train.json                         VoxPopuli it (+ FLEURS it train, + MLS it if asked)
  fleurs_it_validation{,_room}.json  FLEURS it validation, clean and in a simulated room
  fleurs_it_test{,_room}.json        FLEURS it test, the same
  fleurs_<lang>_test.json            other FLEURS languages, clean (--also): forgetting
  rirs_{train,eval}.json             room impulse responses (SLR28), split by room
  noises_{train,eval}.json           noises (SLR28), split by room or file
  audio/, rirs/, noises/             the WAVs; manifests point to them relatively
  PROVENANCE.json                    every source with its commit or sha256, and the hours

Text keeps the source's casing and punctuation (VoxPopuli raw_text, FLEURS
raw_transcription), as v3 writes them. MLS has neither (lowercase, no
punctuation, old literary Italian): training on it teaches the model to drop
them, so it is off unless --mls-hours says otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from common import (
    FLEURS,
    MLS,
    RIRS_SHA256,
    RIRS_SIZE,
    RIRS_URLS,
    SAMPLE_RATE,
    VOXPOPULI,
    absolute,
    clean_text,
    read_manifest,
    sha256,
    simulate_room,
    work_dir,
    write_manifest,
)

SLR28 = "RIRS_NOISES/"
REAL = SLR28 + "real_rirs_isotropic_noises/"


def decode(blob: bytes) -> np.ndarray:
    """Any format libsndfile reads (WAV, FLAC, Opus) to mono float32 at 16 kHz:
    the first channel, resampled if needed."""
    from scipy.signal import resample_poly

    data, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    x = data[:, 0]
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return x


def safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def hf_parquet(repo: str, commit: str, prefix: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=commit)
    found = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not found:
        raise SystemExit(f"{repo}@{commit}: no parquet file under {prefix}")
    return found


def rows(path: Path, columns: list[str], rng: np.random.Generator):
    """A parquet file's rows in a random order, one row group in memory at a
    time."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    for group in rng.permutation(f.num_row_groups):
        table = f.read_row_group(int(group), columns=columns)
        for i in rng.permutation(table.num_rows):
            yield table.slice(int(i), 1).to_pylist()[0]


def take(source: tuple[str, str], files: list[str], cols: tuple[str, str, str], name: str,
         out: Path, rng: np.random.Generator, hours: float | None,
         seconds: tuple[float, float] | None) -> list[dict]:
    """Rows of `files` written as WAVs under audio/<name>/: up to `hours`
    (spread evenly over the files, rows drawn at random), only those whose
    length is within `seconds`. `cols` is (id, audio, text)."""
    from huggingface_hub import hf_hub_download

    repo, commit = source
    key, audio, text_col = cols
    per_file = hours * 3600 / len(files) if hours else math.inf
    kept: list[dict] = []
    folder = out / "audio" / name
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for file in files:
            local = Path(hf_hub_download(repo, file, repo_type="dataset", revision=commit,
                                         local_dir=tmp))
            got = 0.0
            for row in rows(local, [key, audio, text_col], rng):
                if got >= per_file:
                    break
                text = clean_text(row[text_col] or "")
                blob = (row[audio] or {}).get("bytes")
                if not text or not blob:
                    continue
                x = decode(blob)
                duration = len(x) / SAMPLE_RATE
                if seconds and not seconds[0] <= duration <= seconds[1]:
                    continue
                stem = safe(Path(str(row[key])).stem)
                sf.write(folder / f"{stem}.wav", x, SAMPLE_RATE, subtype="PCM_16")
                kept.append({"audio_filepath": f"audio/{name}/{stem}.wav",
                             "duration": round(duration, 3), "text": text})
                got += duration
            local.unlink()
            print(f"  {name}: {file}: {got / 3600:.2f} h", flush=True)
    return kept


def download_rirs(directory: Path) -> Path:
    """SLR28's zip, checked against the pinned sha256 before it is opened."""
    target = directory / "rirs_noises.zip"
    for url in RIRS_URLS:
        print(f"  {url}", flush=True)
        try:
            with urllib.request.urlopen(url) as response, target.open("wb") as f:
                shutil.copyfileobj(response, f, 1 << 20)
        except OSError as error:
            print(f"  failed: {error}")
            continue
        if target.stat().st_size == RIRS_SIZE and sha256(target) == RIRS_SHA256:
            return target
        print("  wrong size or sha256, next mirror")
    raise SystemExit(f"no copy of rirs_noises.zip matched sha256 {RIRS_SHA256}")


def rirs_and_noises(out: Path, simulated: int, rng: np.random.Generator) -> dict[str, list]:
    """SLR28's real RIRs (RWCP, REVERB 2014, AIR), a sample of the simulated
    small and medium rooms (1-30 m, a source within 5 m: a living room to an
    open space), the isotropic noises recorded in the real rooms and the
    point-source noises, each as mono 16 kHz under rirs/ and noises/. One room
    in ten (by a hash of its name, so the split does not move) goes to the
    evaluation sets only: the simulated room of FLEURS validation and test
    never reuses a response or a noise training saw."""
    manifests: dict[str, list] = {k: [] for k in
                                  ("rirs_train", "rirs_eval", "noises_train", "noises_eval")}
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(download_rirs(Path(tmp))) as z:
        names = z.namelist()
        picks: list[tuple[str, str, str]] = []  # (member, room, kind)
        for line in z.read(REAL + "rir_list").decode().splitlines():
            room = line.split("--room-id ")[1].split()[0]
            picks.append((line.split()[-1], room, "rirs"))
        for line in z.read(REAL + "noise_list").decode().splitlines():
            room = line.split("--room-linkage ")[1].split()[0]
            picks.append((line.split()[-1], room, "noises"))
        sim = sorted(n for n in names if n.endswith(".wav") and (
            "/smallroom/" in n or "/mediumroom/" in n))
        for i in sorted(rng.choice(len(sim), size=min(simulated, len(sim)), replace=False)):
            parts = sim[i].split("/")
            picks.append((sim[i], f"{parts[-3]}/{parts[-2]}", "rirs"))
        for n in names:
            if n.startswith(SLR28 + "pointsource_noises/") and n.endswith(".wav"):
                picks.append((n, Path(n).stem, "noises"))
        for member, room, kind in picks:
            x = decode(z.read(member))
            split = "eval" if zlib.crc32(room.encode()) % 10 == 0 else "train"
            stem = safe(member[len(SLR28):].removesuffix(".wav"))
            sf.write(out / kind / f"{stem}.wav", x, SAMPLE_RATE, subtype="PCM_16")
            manifests[f"{kind}_{split}"].append(
                {"audio_filepath": f"{kind}/{stem}.wav",
                 "duration": round(len(x) / SAMPLE_RATE, 3)})
    return manifests


def room_copy(out: Path, source: str, rirs: list[dict], noises: list[dict], seed: int,
              snr: tuple[float, float], gain: tuple[float, float]) -> list[dict]:
    """Every utterance of a manifest in a simulated room, from the evaluation
    RIRs and noises; the generator of utterance i is seeded by (seed, i), so
    the set is the same on every run and for every model evaluated on it."""
    src = out / f"{source}.json"
    name = f"{source}_room"
    folder = out / "audio" / name
    folder.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}

    def load(row: dict) -> np.ndarray:
        path = str(out / row["audio_filepath"])
        if path not in cache:
            cache[path] = sf.read(path, dtype="float32")[0]
        return cache[path]

    result = []
    for i, row in enumerate(read_manifest(src)):
        rng = np.random.default_rng([seed, i])
        rir = load(rirs[int(rng.integers(len(rirs)))])
        noise = load(noises[int(rng.integers(len(noises)))])
        speech = sf.read(absolute(src, row), dtype="float32")[0]
        y = simulate_room(speech, rir, noise, rng, snr, gain)
        stem = Path(row["audio_filepath"]).stem
        sf.write(folder / f"{stem}.wav", y, SAMPLE_RATE, subtype="PCM_16")
        result.append({**row, "audio_filepath": f"audio/{name}/{stem}.wav"})
    return result


def total_hours(rows: list[dict]) -> float:
    return round(sum(r["duration"] for r in rows) / 3600, 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=work_dir("data"))
    p.add_argument("--voxpopuli-hours", type=float, default=60.0,
                   help="cap on VoxPopuli it train, in hours (default 60: ~7 GB of WAV)")
    p.add_argument("--mls-hours", type=float, default=0.0,
                   help="MLS it train, in hours; lowercase without punctuation (default 0)")
    p.add_argument("--no-fleurs-train", action="store_true",
                   help="leave FLEURS it train (~10 h) out of train.json")
    p.add_argument("--min-seconds", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="longer training utterances are skipped (train.py's --max-duration)")
    p.add_argument("--simulated-rirs", type=int, default=4000,
                   help="simulated RIRs sampled from SLR28's small and medium rooms")
    p.add_argument("--also", default="en_us",
                   help="other FLEURS languages, comma separated, whose test set is "
                        "written clean to measure forgetting ('' for none)")
    p.add_argument("--snr", type=float, nargs=2, default=(5.0, 25.0),
                   help="SNR range of the room sets, dB (train.py's default too)")
    p.add_argument("--gain", type=float, nargs=2, default=(-25.0, 0.0),
                   help="gain range of the room sets, dB (train.py's default too)")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    out: Path = args.out
    for sub in ("audio", "rirs", "noises"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    span = (args.min_seconds, args.max_seconds)
    provenance: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "sources": [],
                        "manifests": {}}

    def record(source: tuple[str, str], files: list[str], licence: str) -> None:
        provenance["sources"].append({"repo": source[0], "commit": source[1],
                                      "files": files, "licence": licence})

    print("FLEURS it", flush=True)
    fleurs_cols = ("path", "audio", "raw_transcription")
    fleurs: dict[str, list] = {}
    for split in ("train", "validation", "test"):
        files = hf_parquet(*FLEURS, f"parquet-data/it_it/{split}-")
        fleurs[split] = take(FLEURS, files, fleurs_cols, f"fleurs_it_{split}", out, rng,
                                None, span if split == "train" else None)
        record(FLEURS, files, "CC-BY-4.0")
    for lang in filter(None, args.also.split(",")):
        files = hf_parquet(*FLEURS, f"parquet-data/{lang}/test-")
        rows_ = take(FLEURS, files, fleurs_cols, f"fleurs_{lang}_test", out, rng, None, None)
        write_manifest(out / f"fleurs_{lang}_test.json", rows_)
        record(FLEURS, files, "CC-BY-4.0")

    print("VoxPopuli it", flush=True)
    files = hf_parquet(*VOXPOPULI, "it/train-")
    train = take(VOXPOPULI, files, ("audio_id", "audio", "raw_text"), "voxpopuli_it", out,
                    rng, args.voxpopuli_hours, span)
    record(VOXPOPULI, files, "CC0-1.0")
    if args.mls_hours > 0:
        print("MLS it", flush=True)
        files = hf_parquet(*MLS, "italian/train-")
        mls = take(MLS, files, ("id", "audio", "transcript"), "mls_it", out, rng,
                      args.mls_hours, span)
        train += mls
        record(MLS, files, "CC-BY-4.0")
    if not args.no_fleurs_train:
        train += fleurs["train"]
    rng.shuffle(train)
    write_manifest(out / "train.json", train)
    for split in ("validation", "test"):
        write_manifest(out / f"fleurs_it_{split}.json", fleurs[split])

    print("SLR28 room impulse responses and noises", flush=True)
    room = rirs_and_noises(out, args.simulated_rirs, rng)
    for name, rows_ in room.items():
        write_manifest(out / f"{name}.json", rows_)
    provenance["sources"].append({"url": RIRS_URLS[0], "sha256": RIRS_SHA256,
                                  "licence": "Apache-2.0 (OpenSLR SLR28); point-source "
                                             "noises public domain (Freesound)"})

    print("FLEURS it in a simulated room", flush=True)
    for split in ("validation", "test"):
        name = f"fleurs_it_{split}"
        write_manifest(out / f"{name}_room.json",
                       room_copy(out, name, room["rirs_eval"], room["noises_eval"], args.seed,
                                 tuple(args.snr), tuple(args.gain)))

    for manifest in sorted(out.glob("*.json")):
        if manifest.name != "PROVENANCE.json":
            rows_ = read_manifest(manifest)
            provenance["manifests"][manifest.stem] = {"rows": len(rows_),
                                                      "hours": total_hours(rows_)}
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for name, info in provenance["manifests"].items():
        print(f"{name:28} {info['rows']:7} rows {info['hours']:7.2f} h")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
