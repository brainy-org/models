"""WER and CER of a .nemo on FLEURS it test, clean and in the simulated room.

    python evaluate.py --data DIR [--model base | PATH.nemo] [--out results.json]

The room set is written once by prepare.py from its own seed and from RIRs and
noises training never saw, so the baseline and every tuned model are scored
on the same files. Reference and hypothesis go through the same normalisation
(lower case, no punctuation: common.normalize_for_wer); greedy decoding, as
parakeet.cpp decodes. Writes a JSON of the scores and, beside it, a JSON lines
file of every hypothesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    BASE_COMMIT,
    BASE_REPO,
    absolute,
    fetch_base,
    normalize_for_wer,
    read_manifest,
    sha256,
    work_dir,
)

SETS = ("fleurs_it_test", "fleurs_it_test_room", "fleurs_en_us_test")


def text_of(hypothesis) -> str:
    # transcribe() gives Hypothesis objects (2.x) or strings.
    return getattr(hypothesis, "text", hypothesis) or ""


def score(pairs: list[tuple[str, str]]) -> dict:
    import editdistance

    words = chars = word_errors = char_errors = 0
    for ref, hyp in pairs:
        r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
        word_errors += editdistance.eval(r.split(), h.split())
        char_errors += editdistance.eval(r, h)
        words += len(r.split())
        chars += len(r)
    return {"wer": round(100 * word_errors / max(words, 1), 2),
            "cer": round(100 * char_errors / max(chars, 1), 2),
            "utterances": len(pairs), "words": words}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True, help="prepare.py's output folder")
    p.add_argument("--model", default="base",
                   help="'base' (the pinned nvidia/parakeet-tdt-0.6b-v3) or a .nemo")
    p.add_argument("--sets", nargs="+", default=list(SETS),
                   help="manifests of DIR, by name; a missing one is skipped")
    p.add_argument("--out", type=Path, help="default: <work>/eval/<model>.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, help="first N utterances of each set (a smoke test)")
    args = p.parse_args()

    import torch
    from nemo.collections.asr.models import ASRModel

    path = fetch_base() if args.model == "base" else Path(args.model)
    name = "base" if args.model == "base" else path.stem
    out = args.out or work_dir("eval") / f"{name}.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASRModel.restore_from(str(path), map_location=device)
    model.eval()

    result = {"model": str(path), "sha256": sha256(path),
              "base": {"repo": BASE_REPO, "commit": BASE_COMMIT},
              "normalisation": "NFKC, lower case, punctuation and symbols to spaces",
              "decoding": "greedy_batch", "sets": {}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".hyps.jsonl").open("w", encoding="utf-8") as hyps:
        for set_name in args.sets:
            manifest = args.data / f"{set_name}.json"
            if not manifest.exists():
                print(f"{set_name}: no {manifest}, skipped")
                continue
            rows = read_manifest(manifest)[: args.limit]
            audio = [absolute(manifest, row) for row in rows]
            with torch.inference_mode():
                output = model.transcribe(audio, batch_size=args.batch_size, verbose=False)
            if isinstance(output, tuple):  # older NeMo: (best, all)
                output = output[0]
            pairs = [(row["text"], text_of(h)) for row, h in zip(rows, output, strict=True)]
            for row, (_, hyp) in zip(rows, pairs, strict=True):
                hyps.write(json.dumps({"set": set_name, "audio": row["audio_filepath"],
                                       "ref": row["text"], "hyp": hyp},
                                      ensure_ascii=False) + "\n")
            result["sets"][set_name] = score(pairs)
            print(set_name, result["sets"][set_name], flush=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
