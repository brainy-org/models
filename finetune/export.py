"""A .nemo from one of train.py's Lightning checkpoints.

    python export.py CHECKPOINT.ckpt OUT.nemo

The base .nemo (pinned) gives the configuration and the tokenizer, the
checkpoint the weights: the same model train.py would have written at that
step. For a run whose session ended before its own save_to (the sixth was
killed by the out-of-memory killer after its last step), and for the best
checkpoint by validation WER, which train.py keeps beside the last one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import BASE_COMMIT, BASE_REPO, BASE_SHA256, fetch_base, sha256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()

    import torch
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.restore_from(str(fetch_base()), map_location="cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Lightning's keys are the module's own: a missing or extra one is an
    # error, never a model half tuned.
    model.load_state_dict(state["state_dict"], strict=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(args.out))
    record = {"base": {"repo": BASE_REPO, "commit": BASE_COMMIT, "sha256": BASE_SHA256},
              "checkpoint": str(args.checkpoint), "step": state.get("global_step"),
              "nemo": str(args.out), "sha256": sha256(args.out)}
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
