# Fine-tuning parakeet-tdt-0.6b-v3 for Italian across a room

A recipe, run on Kaggle's free GPUs, that tunes NVIDIA's
`parakeet-tdt-0.6b-v3` (the ASR of the `parakeet` bundle: arvel, logbook,
brainy-worker) on Italian heard the way arvel hears it: from a phone or an iPad
on a table, a few metres away, in a room. The result is a `.nemo` and its
q8_0 GGUF for parakeet.cpp; the same tokenizer, vocabulary and architecture
as the original, so libparakeet v0.6.1 loads it in its place. Nothing here is
mirrored: a tuned model becomes a bundle only once it wins on the owner's own
recordings (below, "What next").

## Why the room, and not more Italian

v3 was trained on Granary and NVIDIA's own ASR set, which very likely hold the
public Italian corpora already (VoxPopuli, MLS, perhaps FLEURS: its clean
scores may flatter the baseline, which the comparison shares): training
again on them, clean, teaches it little.
What it has seen less of is speech a few metres from a phone's microphone:
reverberation, the room's noise, a low level. So the training audio is public
Italian speech put into a simulated room on the fly: convolved with a real or
simulated room impulse response, with a noise added at 5-25 dB SNR, its level
lowered by up to 25 dB, each with a probability, so about a sixth of it stays
clean (a model that forgets clean speech is no better). No private recording
leaves the owner's computer: the owner's own audio is used only to evaluate,
locally, afterwards.

## What it uses, pinned

| What | Pin | Licence |
|---|---|---|
| `nvidia/parakeet-tdt-0.6b-v3` (`.nemo`) | commit `541d1f99`, sha256 `3cbdc858…` (`common.py`) | CC-BY-4.0 |
| VoxPopuli it, `facebook/voxpopuli` (train, 60 h by default; `raw_text`) | commit `42f01879` | CC0-1.0 (European Parliament recordings, reuse with acknowledgement) |
| FLEURS it, `google/fleurs` (`it_it` train for training; validation and test held out; `en_us` test for forgetting) | commit `70bb2e84` | CC-BY-4.0 |
| MLS it, `facebook/multilingual_librispeech` (off by default) | commit `2e83e618` | CC-BY-4.0 |
| OpenSLR SLR28 `rirs_noises.zip`: 325 real RIRs (RWCP, REVERB 2014, AIR), 4,000 of the simulated small and medium rooms, 92 isotropic room noises, 843 point-source noises | sha256 `3b50cfde…` | Apache-2.0 (OpenSLR); point-source noises public domain (Freesound) |
| NeMo | `nemo_toolkit[asr,cu12]==2.7.3` (the version parakeet.cpp's converter is validated with), torch 2.8.0 + CUDA 12.8, Python 3.12 | Apache-2.0 |
| parakeet.cpp's converter (`convert.sh`) | commit `5c9b0fbb` (tag v0.6.1, the `libparakeet` bundle's release) | MIT |

The datasets are read from their parquet files at a commit (no loading
script, no `datasets` library); OpenSLR publishes no checksum, so the
SLR28 sha256 is the one two Hugging Face copies of the file record, and
`prepare.py` stops if the download differs. Common Voice is not used: it left
Hugging Face in 2025 for Mozilla's Data Collective, which needs an account
and its terms accepted.

**Text.** v3 writes punctuation and capitals, so the targets keep them:
VoxPopuli's `raw_text` and FLEURS's `raw_transcription`, only typographic
quotes folded. MLS's transcripts have neither (and are old literary Italian):
training on them would teach the model to drop both, so MLS is off unless
`--mls-hours` asks for it. An utterance whose text the tokenizer can only
write as `<unk>` is dropped (`train.py`): the vocabulary does not change.
WER and CER are computed on lower case without punctuation, the same for
every model (`common.normalize_for_wer`).

**The room sets.** `prepare.py` splits the RIRs and noises by room (one in
ten, by a hash of its name): training never hears the rooms of the
evaluation, and FLEURS validation and test are written once more in those
rooms with a fixed seed (`*_room.json`), so the baseline and every tuned model
are scored on the same files.

## On Kaggle

Everything heavy runs there; the repository is private, so the scripts go as
a private dataset: on kaggle.com, *Datasets → New Dataset*, upload this
folder's files, name it e.g. `parakeet-finetune`. Kaggle mounts what a
notebook attaches under `/kaggle/input/`: run `!ls -R /kaggle/input | head`
and adjust the paths below. A first Python cell names the scripts' folder,
which the `!` lines expand: `SRC = "/kaggle/input/parakeet-finetune"`.

Limits as of November 2026 (they move; the account's settings show the
quota): a session runs 12 h at most, the GPU quota is reported at about 30 h a week,
`/kaggle/working` is the output kept with a version (20 GB). The GPU to pick
is **T4 x2** (16 GB each, no bf16, so fp16 mixed precision). Not the P100:
Kaggle's PyTorch dropped its sm_60 (Kaggle/docker-python#1546).

**1. Data** (a CPU notebook, *Internet on*, the scripts attached):

```
!python $SRC/prepare.py --out /kaggle/working/data
```

Then *Save Version → Save & Run All*. About 45-90 min (an estimate: ~20 GB of
parquet and 1.3 GB of zip downloaded, decoded and written); it prints the
hours of each manifest and the folder's size (~11 GB with the defaults). The
version's output is the data: attach it to the next notebooks.

**2. Training** (a notebook with *Accelerator: GPU T4 x2*, *Internet on*, the
scripts and the data attached). The first cell builds a Python 3.12
environment (5-10 min):

```
!pip install -q uv
!uv venv -q -p 3.12 /tmp/venv
!uv pip install -q -p /tmp/venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
!uv pip install -q -p /tmp/venv/bin/python -r $SRC/requirements.txt
```

then:

```
!MALLOC_ARENA_MAX=2 /tmp/venv/bin/python $SRC/train.py --data /kaggle/input/<data>/data --out /kaggle/working/run
```

and *Save Version → Save & Run All* (a background run: an interactive
session stops when idle). It prints a line every 50 steps and validates every
500 on FLEURS it validation, clean and in the room (`run/logs/*/metrics.csv`).
After `--hours` (10.5) it saves `run/checkpoints/resume.ckpt` and stops; at
the end of every session it writes `run/parakeet-tdt-0.6b-v3-it-room.nemo`
(where it got to) and `run/TRAINED.json` (the pins, the arguments, the step).

**3. Resume**: a new version of the same notebook, with the previous version's
output attached (*Add Input → Your Work*, the notebook), and

```
!MALLOC_ARENA_MAX=2 /tmp/venv/bin/python $SRC/train.py --data /kaggle/input/<data>/data --out /kaggle/working/run \
    --resume /kaggle/input/<previous>/run/checkpoints/resume.ckpt
```

with the same `--max-steps` (the cosine schedule was made for it). Weights,
optimizer, scheduler and step come back; the epoch's order of utterances does
not (a new shuffle), which does not matter here.

If the session ended without the `.nemo` (killed after its last step, or
the best validation was an earlier step), make it from a checkpoint:

```
!/tmp/venv/bin/python $SRC/export.py /kaggle/input/<previous>/run/checkpoints/resume.ckpt \
    /kaggle/working/run/parakeet-tdt-0.6b-v3-it-room.nemo
```

**4. Evaluation** (the same environment, one GPU is enough):

```
!/tmp/venv/bin/python $SRC/evaluate.py --data /kaggle/input/<data>/data --model base --out /kaggle/working/eval/base.json
!/tmp/venv/bin/python $SRC/evaluate.py --data /kaggle/input/<data>/data \
    --model /kaggle/working/run/parakeet-tdt-0.6b-v3-it-room.nemo --out /kaggle/working/eval/tuned.json
```

WER and CER on FLEURS it test clean, in the room, and FLEURS en_us test
(forgetting); each JSON has a `.hyps.jsonl` beside it with every hypothesis.
About 5-10 min per model (an estimate).

**5. GGUF** (the same environment, CPU is enough):

```
!PYTHON=/tmp/venv/bin/python bash $SRC/convert.sh /kaggle/working/run/parakeet-tdt-0.6b-v3-it-room.nemo
```

It clones parakeet.cpp at the pinned commit, converts to q8_0, checks every
`parakeet.*` key (tokenizer pieces included) and every tensor's name and shape
against the GGUF of the mirror's `parakeet` bundle, and prints the new file's
sha256 and size (~940 MB, like the original).

## Memory and time on a T4

The model holds 0.6 B parameters: weights and gradients in fp32 take 5 GB,
AdamW's state 5 GB more; the joint's logits (8,198 outputs per frame and
token, cast to fp32 for the TDT loss) are the largest activation. The
defaults keep the optimizer state in 8 bits (bitsandbytes, ~1.3 GB), the
gradients in DDP's buckets (no second copy, 2.5 GB), cap utterances at 15 s,
and take 2 per GPU with joint sub-batches of 1, 8 batches a step. The first
run, at 4 per GPU, sub-batches of 2 and 20 s, ran out of memory in the TDT
loss at its first step (13.5 of 14.6 GB in use, 1.8 GB more asked); the
second, at 2 per GPU and sub-batches of 1, a few minutes in, on a long
utterance (13.8 GB in use, 0.95 GB more asked). Still out of memory:
`--freeze-encoder-layers 8`, then `--batch-size 1 --accumulate 16`. With `--optimizer adamw` (no bitsandbytes)
start from `--batch-size 1 --accumulate 16`. A rank that fails ends the run
within 10 minutes (the DDP timeout), not after NCCL's 30.

Host memory is the other limit. Lightning writes a checkpoint by
serializing it into memory and copying it out, twice its ~4 GB: the third run
was killed by the kernel's out-of-memory killer at its first checkpoint
(step 250), so `train.py` writes it straight to the file. The fourth run
lost RAM steadily instead, 18 GB free at the start and 1.7 GB at step 500,
where it was killed. The fifth, with unpinned loaders and
`MALLOC_ARENA_MAX=2`, lost it the same way while the ranks' own memory stayed
flat after step 150: the data loader workers, forks of a 6 to 11 GB rank,
copying its pages as they touch them. The loaders now run in the rank
itself (`--num-workers 0`), every rank gives freed memory back to the system
every 50 steps (`malloc_trim`), and the notebook keeps `MALLOC_ARENA_MAX=2`.
Each step line prints the RAM still available, the shared memory, rank 0's
resident memory and its children's (rank 1 and any loaders).

The defaults train 3,000 steps of 32 utterances (~12 s each): ~320 h of
audio, about four passes over the ~70 h of data. On two T4 a step took 6.3 s
on the third and fourth runs: 3,000 steps in about 5.3 h, one session.

## Risks

- **Forgetting.** v3 speaks 25 languages, and arvel may run in another one;
  these 70 h are Italian only. The learning rate is low (2e-5 peak, v3's own
  config ends at 1e-5) with a warm-up, part of the audio stays clean, and
  `evaluate.py` scores FLEURS en_us: a tuned model that loses there is not a
  replacement for every language pack. `--freeze-encoder-layers` and
  `--freeze-decoder` narrow what can move.
- **fp16 with the TDT loss.** The loss runs in fp32 (NeMo casts the joint's
  output for numba), gradients are clipped at 1.0 and the AMP scaler skips a
  step with an inf or a NaN; a run of skipped steps (the learning rate stops
  moving, the loss is `nan`) means a lower `--lr`.
- **Short commands.** FLEURS and VoxPopuli are sentences of 5-20 s; arvel
  hears commands of one or two seconds. The acoustic adaptation should carry
  over; only the owner's recordings say whether it does.
- **The simulation is harsher than the satellite.** The page's audio has been
  through the browser's echo cancellation and noise suppression before
  arvel's ASR gets it; the simulated noise has not.
- **Unverified here.** Nothing of this was run before it was written: the
  NeMo, Lightning and bitsandbytes calls were checked against their sources
  at the pinned versions, the Kaggle limits against Kaggle's notes. The first
  session will show what does not hold.

## Licences

The tuned model is an adaptation of `nvidia/parakeet-tdt-0.6b-v3`, under
CC-BY-4.0: wherever it goes, it says so ("based on NVIDIA's
parakeet-tdt-0.6b-v3, CC-BY-4.0, fine-tuned by brainy-org on Italian speech
in simulated rooms; changes were made"), and a bundle of it carries that in
its `NOTICE.md`. The training data (CC0 and CC-BY-4.0 speech, Apache-2.0 and
public domain RIRs and noises) is used, not redistributed; the CC-BY sources
(FLEURS, MLS if used) are credited in the same notice, from
`data/PROVENANCE.json`.

## The first run (2026-11-10)

No gain: the recipe as it stands is not worth mirroring. Two T4, the
defaults, 3,000 steps (4.9 h); scored by `evaluate.py` (WER %, normalised):

| | it clean | it room | en_us |
|---|---|---|---|
| base | 2.99 | 3.48 | 5.45 |
| best validation (step 2000) | 2.88 | 3.58 | 9.17 |
| last (step 3000) | 2.95 | 3.45 | 93.06 |

- On Italian the differences are within noise (0.1 points is about 20 of
  21,249 words). The simulated room costs the base only half a point: v3 is
  already robust to it, and there is little to win where the recipe aims.
- English is forgotten: the best checkpoint loses 3.7 points and drops the
  start of sentences; the last returns an empty transcript for almost every
  English utterance (validation in Italian rose from 6.3 to 9.4 % at step
  2500 too). Fine-tuning on Italian only, with no other language in the
  batches, wears away what it does not see.
- The GGUF that `convert.sh` wrote fails its own check against the mirror's:
  `parakeet.arch` differs, and `parakeet.encoder.use_bias` is only in the new
  file. The mirror's GGUF was not made by the converter at v0.6.1 as pinned
  here; to be found before any tuned model could load as the same one.

Worth trying before another run: measure the base on the owner's own
recordings first (below), to find where it fails in the real room; if the
failures are there, mix other languages into the batches (English at least)
and stop at the best validation.

## The base on the owner's recordings (2026-11-10)

The room is hard and the base does no worse than Whisper there: there is not
enough of it to train on, and it is the only real measure, so it stays the
evaluation set. 406 phrases (25.3 min) cut by Silero's VAD from
`arvel chat --record-room` files of 2026-11-07 and 2026-11-10 (a family at
home, children's voices, people talking over each other), drafted by the
mirror's parakeet GGUF (through libparakeet, `it`, no primer) and Whisper
large-v3, corrected by ear by the owner; 121 phrases (5.9 min) were
unintelligible to a person too and are left out. Scored with this recipe's
normalisation, on 285 phrases, 2,693 words:

| | WER % | CER % |
|---|---|---|
| parakeet (base) | 25.8 | 15.5 |
| Whisper large-v3 | 34.4 | 21.4 |

- **By length** (parakeet, the 246 phrases the models disagreed on): under
  2 s 80 %, 2-5 s 38 %, 5-10 s 18 %, over 10 s 16 %. Short phrases are
  where it fails, and they are arvel's commands. Measured without arvel's
  language primer (Italian speech put before each phrase live, so a short
  one is not read in another language); the number to compare a tuned model
  with is this one, same setup.
- **With the primer** (arvel's, synthesized as `chat` makes it, put before
  each clip as `src/listen/primer.rs` does): under 2 s 61 % (from 80),
  2-5 s 35 % (38), but 5-10 s 31 % (18) and over 10 s 24 % (16); no phrase
  in English left (from 20). By WER it helps short phrases and hurts long
  ones, but the references were edited from parakeet's draft without the
  primer, which favours that draft: in a blind comparison by ear of the 66
  clips of 5 s or more where the two differ (order drawn per clip, which is
  which hidden), the owner preferred the one without 15 times, the one with
  13, and found them equal 38 times. The primer costs long phrases nothing
  that a listener hears; the WER gap over 5 s is the prefill's.
- **English**: 21 phrases came out mostly in English (v3 picks the language
  itself, `it` notwithstanding).
- **The review's bias**: each reference was prefilled with parakeet's draft
  (51 of 246 kept as is, 30 taken from Whisper), so the comparison with
  Whisper leans towards parakeet by an amount not measured; the blind
  comparison above shows the pull is large enough to make a difference
  appear where a listener hears none. A comparison of two models on this set
  is made blind (`ab.py` beside it), not by WER on these references.
- **The audio itself**: 121 of 367 phrases were unintelligible to the owner
  too. Their speech is only ~4 dB quieter than the clear ones (-18 vs -14
  dBFS) at a similar distance from the noise floor (28 vs 32 dB): distance,
  reverberation, overlapping voices and children, not level or noise. No
  model does better than a listener there; only the microphone's place does.
- **The name**: 48 occurrences, about two thirds misspelled ("harvel",
  "albert", "well"). arvel matches it by sound and searches the beam's other
  hypotheses (arvel's `src/listen/wake.rs`), so most would still wake it.
- Recordings of quiet commands score 1-15 %; conversation across the room
  25-30 %.

The set (clips, references, the scripts that built it) lives beside the
recordings on the owner's computer, never in a repository, and is deleted
with them once a tuned model has been measured on it.

## What next

1. ~~Evaluate the base locally on the owner's own recordings~~: done, above.
   A tuned model is scored the same way, on the same clips, through
   libparakeet. That audio never leaves the computer.
2. Only if the tuned model wins there (and does not lose on clean speech and
   on the other languages arvel uses): mirror it as a new bundle with
   `mirror.py` (the tuned GGUF from wherever it is kept, by sha256, with its
   `NOTICE.md`), then move the consumers' pins (`../README.md`, "Adding or
   changing a model").

## Files

| File | What |
|---|---|
| `common.py` | the pins, the folders, manifests, text normalisation, the room simulation of the evaluation sets |
| `prepare.py` | data, RIRs and noises to WAVs and manifests (CPU) |
| `train.py` | the fine-tune, resumable across sessions (GPU) |
| `export.py` | a `.nemo` from a checkpoint (`resume.ckpt`, the last step; `best.ckpt`, the best validation in the room) |
| `evaluate.py` | WER and CER of a `.nemo`, clean and in the room |
| `convert.sh` | the q8_0 GGUF, checked against the mirror's |
| `requirements.txt` | the environment of `train.py`, `evaluate.py` and `convert.sh` |
