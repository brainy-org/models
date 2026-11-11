"""Fine-tune parakeet-tdt-0.6b-v3 on Italian heard across a room, with NeMo.

    python train.py --data DIR [--out DIR] [--resume CKPT] [--max-steps 3000] ...

DIR is prepare.py's output. Every GPU of the machine is used (two T4 on
Kaggle, DDP), in fp16 mixed precision: the T4 has no bf16. The audio is
reverberated, noised and attenuated on the fly by NeMo's augmentors
(impulse, noise, gain), each with a probability, so part of it stays clean.

A Kaggle session ends at 12 h: training stops by itself after --hours, writes
checkpoints/resume.ckpt (weights, optimizer, scheduler, step) and the .nemo of
where it got to, both in OUT. The next session passes that resume.ckpt to
--resume, with the same --max-steps, and goes on from the same step. The
tokenizer, the vocabulary and the architecture are the base model's,
untouched: parakeet.cpp converts the result like the original.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

# Before torch is imported: fewer out-of-memory errors from fragmentation with
# utterances of every length.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from common import (
    BASE_COMMIT,
    BASE_REPO,
    BASE_SHA256,
    absolute,
    fetch_base,
    read_manifest,
    work_dir,
    write_manifest,
)

# Lightning starts the other GPUs' processes by running this script again with
# LOCAL_RANK set: only the first one checks the download and writes files.
MAIN = "LOCAL_RANK" not in os.environ
NEMO_NAME = "parakeet-tdt-0.6b-v3-it-room.nemo"


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True, help="prepare.py's output folder")
    p.add_argument("--out", type=Path, default=work_dir("run"))
    p.add_argument("--resume", type=Path, help="resume.ckpt of an earlier session")
    p.add_argument("--hours", type=float, default=10.5,
                   help="stop and save after this long (a Kaggle session is 12 h)")
    p.add_argument("--max-steps", type=int, default=3000,
                   help="optimizer steps in all; keep it the same when resuming")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="peak learning rate (v3's own config ends at 1e-5)")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=300, help="warm-up steps")
    p.add_argument("--optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit",
                   help="adamw8bit (bitsandbytes) keeps the optimizer state in ~1.3 GB "
                        "instead of ~5 GB")
    # A batch of 4 with sub-batches of 2 ran out of a T4's 16 GB in the TDT
    # loss at the first step (the joint's fp32 output and its gradient).
    p.add_argument("--batch-size", type=int, default=2, help="utterances per GPU")
    p.add_argument("--accumulate", type=int, default=8,
                   help="batches per optimizer step (32 utterances a step on two GPUs)")
    p.add_argument("--fused-batch-size", type=int, default=1,
                   help="utterances per joint/loss sub-batch (v3 ships 4)")
    # At 20 s a long utterance still ran out of memory on the second run, in
    # the joint (its size grows with frames times tokens).
    p.add_argument("--max-duration", type=float, default=15.0, help="seconds")
    p.add_argument("--devices", type=int, default=0, help="GPUs; 0: all of them")
    p.add_argument("--precision", default="16-mixed",
                   help="16-mixed on T4 and P100; bf16-mixed where the GPU has it")
    # Workers are forks of a rank (6 to 11 GB) that copy its pages as they
    # touch them: with 2 per GPU the fifth run's free RAM went from 18 GB to
    # 1 GB in 500 steps while the ranks' own memory stayed flat.
    p.add_argument("--num-workers", type=int, default=0,
                   help="data loader processes per GPU; 0: in the rank itself")
    p.add_argument("--val-every", type=int, default=500, help="optimizer steps")
    p.add_argument("--ckpt-every", type=int, default=250, help="optimizer steps")
    p.add_argument("--p-reverb", type=float, default=0.6)
    p.add_argument("--p-noise", type=float, default=0.6)
    p.add_argument("--p-gain", type=float, default=0.5)
    p.add_argument("--snr", type=float, nargs=2, default=(5.0, 25.0), help="dB")
    p.add_argument("--gain", type=float, nargs=2, default=(-25.0, 0.0), help="dB")
    p.add_argument("--freeze-encoder-layers", type=int, default=0,
                   help="freeze the subsampling and the first N of 24 conformer layers")
    p.add_argument("--freeze-decoder", action="store_true",
                   help="freeze the prediction network and the joint")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def manifest(model, data: Path, name: str, out: Path) -> str:
    """`name`'s manifest with absolute paths (the data folder is read-only on
    Kaggle), without the utterances whose text the tokenizer can only write
    as <unk>: the vocabulary does not change, so the text must fit it."""
    src, dst = data / f"{name}.json", out / "manifests" / f"{name}.json"
    if MAIN:
        tokenizer = model.tokenizer
        rows, dropped = [], 0
        for row in read_manifest(src):
            if "text" in row and tokenizer.unk_id in tokenizer.text_to_ids(row["text"]):
                dropped += 1
                continue
            rows.append({**row, "audio_filepath": absolute(src, row)})
        write_manifest(dst, rows)
        print(f"{name}: {len(rows)} rows, {dropped} dropped for <unk>", flush=True)
    return str(dst)


def main() -> None:
    args = parse()
    # Before CUDA starts: less fragmentation between utterances of every length.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from datetime import timedelta

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.plugins.io import TorchCheckpointIO
    from lightning.pytorch.strategies import DDPStrategy
    from lightning.pytorch.utilities import rank_zero_info
    from nemo.collections.asr.models import ASRModel

    started = time.monotonic()
    pl.seed_everything(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        from nemo.core.config.optimizers import AdamWParams
        from nemo.core.optim.optimizers import register_optimizer

        register_optimizer("adamw8bit", bnb.optim.AdamW8bit, AdamWParams())

    model = ASRModel.restore_from(str(fetch_base(check=MAIN)), map_location="cpu")
    model.joint.set_fused_batch_size(args.fused_batch_size)

    frozen = []
    if args.freeze_encoder_layers:
        frozen += [model.encoder.pre_encode, *model.encoder.layers[: args.freeze_encoder_layers]]
    if args.freeze_decoder:
        frozen += [model.decoder, model.joint]
    for module in frozen:
        module.requires_grad_(False)

    class DirectCheckpointIO(TorchCheckpointIO):
        """Writes the checkpoint straight to its file. Lightning's own
        serializes it into a BytesIO and copies that out (twice its ~4 GB in
        RAM): the third run was killed by the out-of-memory killer at its
        first checkpoint."""

        def save_checkpoint(self, checkpoint, path, storage_options=None):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            torch.save(checkpoint, partial)
            os.replace(partial, path)

    def meminfo_gb(key: str) -> float:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2**20
        return float("nan")

    libc = ctypes.CDLL("libc.so.6")

    def rss_gb() -> tuple[float, float]:
        """This process's resident memory and its children's together (rank
        1 and the data loaders: shared pages counted in each)."""
        def rss(pid: str) -> float:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
            return 0.0
        me = str(os.getpid())
        children = [d.name for d in Path("/proc").iterdir() if d.name.isdigit()
                    and (d / "stat").exists()
                    and (d / "stat").read_text().rsplit(")", 1)[-1].split()[1] == me]
        return rss(me), sum(rss(c) for c in children)

    class Session(Callback):
        """Stops before Kaggle does (Lightning's own max_time is saved in the
        checkpoint and would count the earlier sessions too), keeps frozen
        modules in eval mode (no dropout, no batch-norm statistics), and prints
        a line every 50 steps instead of a progress bar."""

        def on_train_batch_start(self, trainer, *_):
            for module in frozen:
                module.eval()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            # At the end of an optimizer step only, so no accumulated gradient
            # is lost; every rank saves (Lightning writes from rank 0).
            late = time.monotonic() - started > args.hours * 3600
            boundary = (batch_idx + 1) % args.accumulate == 0
            if trainer.strategy.broadcast(late and boundary) and not trainer.should_stop:
                rank_zero_info(f"{args.hours} h reached at step {trainer.global_step}: saving")
                trainer.save_checkpoint(out / "checkpoints" / "resume.ckpt")
                trainer.should_stop = True
            step = trainer.global_step
            if step % 50 == 0:
                libc.malloc_trim(0)  # freed audio buffers back to the system
            if trainer.is_global_zero and step % 50 == 0 and step != getattr(self, "last", -1):
                self.last = step
                metrics = {k: f"{float(v):.4g}" for k, v in trainer.callback_metrics.items()
                           if k in ("train_loss", "learning_rate", "training_batch_wer")}
                hours = (time.monotonic() - started) / 3600
                main, loaders = rss_gb()
                print(f"step {step}/{args.max_steps} {hours:.2f} h {metrics} "
                      f"RAM free {meminfo_gb('MemAvailable'):.1f} GB, "
                      f"shared {meminfo_gb('Shmem'):.1f} GB, rank 0 {main:.1f} GB, "
                      f"its children {loaders:.1f} GB", flush=True)

    devices = args.devices or torch.cuda.device_count()
    checkpoint = ModelCheckpoint(
        dirpath=out / "checkpoints", filename="resume", monitor=None, save_top_k=1,
        every_n_train_steps=args.ckpt_every, enable_version_counter=False)
    # The weights of the best validation in the room, beside the last step's
    # (the sixth run's WER rose from 6.3 to 8.6 % between steps 2000 and
    # 2500): export.py makes a .nemo of either.
    best = ModelCheckpoint(
        dirpath=out / "checkpoints", filename="best", monitor="fleurs_it_validation_room_val_wer",
        mode="min", save_top_k=1, save_weights_only=True, enable_version_counter=False)
    # A rank that dies leaves the other waiting in a collective: 10 min, not
    # NCCL's 30, of a session's GPU time. Gradients as views of DDP's buckets
    # instead of a copy beside them: 2.5 GB less on each GPU.
    trainer = pl.Trainer(
        accelerator="gpu", devices=devices,
        strategy=DDPStrategy(timeout=timedelta(minutes=10), gradient_as_bucket_view=True) if devices > 1 else "auto",
        sync_batchnorm=devices > 1, precision=args.precision, max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate, gradient_clip_val=1.0,
        val_check_interval=args.val_every * args.accumulate, check_val_every_n_epoch=None,
        num_sanity_val_steps=0, log_every_n_steps=25, enable_progress_bar=False,
        logger=CSVLogger(out, name="logs"), default_root_dir=out, benchmark=False,
        callbacks=[checkpoint, best, Session(), LearningRateMonitor("step")],
        plugins=[DirectCheckpointIO()])
    model.set_trainer(trainer)

    augmentor = {  # applied in this order: the room, then its noise, then the level
        "impulse": {"prob": args.p_reverb,
                    "manifest_path": manifest(model, args.data, "rirs_train", out)},
        "noise": {"prob": args.p_noise,
                  "manifest_path": manifest(model, args.data, "noises_train", out),
                  "min_snr_db": args.snr[0], "max_snr_db": args.snr[1]},
        "gain": {"prob": args.p_gain, "min_gain_dbfs": args.gain[0],
                 "max_gain_dbfs": args.gain[1]},
    }
    # Not pinned: PyTorch caches pinned host blocks by size and never frees
    # them, and audio batches come in every size; the fourth run lost 16 GB of
    # RAM in 500 steps and was killed.
    common = {"sample_rate": 16000, "use_lhotse": False, "use_start_end_token": False,
              "num_workers": args.num_workers, "pin_memory": False}
    model.setup_training_data({
        **common, "manifest_filepath": manifest(model, args.data, "train", out),
        "batch_size": args.batch_size, "shuffle": True, "max_duration": args.max_duration,
        "min_duration": 0.5, "augmentor": augmentor})
    model.setup_multiple_validation_data({
        **common, "batch_size": 8, "shuffle": False, "manifest_filepath": [
            manifest(model, args.data, "fleurs_it_validation", out),
            manifest(model, args.data, "fleurs_it_validation_room", out)]})
    model.setup_optimization({
        "name": args.optimizer, "lr": args.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing", "warmup_steps": args.warmup,
                  "min_lr": args.min_lr}})

    trainer.fit(model, ckpt_path=str(args.resume) if args.resume else None)

    model.save_to(str(out / NEMO_NAME))  # NeMo writes from rank 0 only
    if MAIN:
        record = {"base": {"repo": BASE_REPO, "commit": BASE_COMMIT, "sha256": BASE_SHA256},
                  "step": trainer.global_step, "max_steps": args.max_steps,
                  "args": {k: str(v) for k, v in vars(args).items()},
                  "data": json.loads((args.data / "PROVENANCE.json").read_text())}
        (out / "TRAINED.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"step {trainer.global_step}: {out / NEMO_NAME}", flush=True)


if __name__ == "__main__":
    main()
