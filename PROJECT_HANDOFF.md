# MultiModalVC — Project Handoff

**Audio-visual pathological → healthy voice conversion (the "MelVC" model).**

Written for an agent/engineer picking up this project cold. It covers what the system
does, why it is built this way, where every important file lives, what has been tried,
and what the numbers currently say.

Last updated: 2026-09-14.

---

## 1. The problem and the objective

**Primary objective: convert pathological-sounding speech into healthy-sounding speech.**

The speakers in the target population have a speech pathology (the in-domain corpus is
German, recorded with synchronised video). The model takes their speech — plus the video
of their face — and produces speech that sounds healthy while preserving what was said.

**Speaker similarity is a secondary, instrumental goal, not the objective.** The model is
conditioned on a target speaker embedding, so it performs voice conversion toward a chosen
healthy voice. That machinery exists because it is the mechanism for producing healthy
speech, not because matching a particular target voice is the point. When trading off
metrics, intelligibility (WER) and healthy-sounding output matter more than SECS.

Three evaluation domains, always reported separately:

| Set | Content | Size | Why it exists |
|---|---|---|---|
| **in-domain test** | held-out pathological German (RECS) | 1,608 utts | the actual task |
| **unseen German** | 2 speakers never trained on | 240 utts | generalization |
| **unseen English** | 2 speakers never trained on | 187 utts | cross-lingual generalization |

---

## 2. Architecture

Model class: `MMS_Speech_NoLLM_MelVC`, registered with fairseq as
`"MMS_Speech_NoLLM_MelVC"`.

```
 video (mp4, 25 fps) ──► AV-HuBERT (large_vox_iter5) ──┐
                                                        ├─► fusion (concat)
 input audio (16 kHz) ─► Whisper-medium encoder ─► afeat_1d_conv ──┘
                                                        │
                                                        ▼
                                            Q-Former compression
                                        (query_tokens, avfeat_to_llm)
                                                        │
                                     ┌──────────────────┴─────────────────┐
                                     │                                    │
                            duration predictor                    proj1 → ln1
                          (predicts output length N)                      │
                                     │                        transposed-conv upsample
                                     └───────────► N ────────► + linear interpolate
                                                                          │
                                                                    proj2 → ln2
                                                                          │
                                               conformer "L" (512-dim), with
                                          INTERLEAVED cross-attention blocks:
                                        speaker x-attn (spk_emb) + content x-attn
                                                                          │
                                                                    ln3 → mel_head
                                                                    (Linear 512→80)
                                                                          │
                                                              80-band BigVGAN mels
                                                                          │
                                                   (separate, frozen) BigVGAN vocoder
                                                                          │
                                                                       waveform
```

### Key design points

- **The model emits mels, not waveforms.** `forward_speech` stops at `mel_head`. The
  parent class builds a HiFi-GAN vocoder in `__init__`; MelVC deletes it
  (`modelSpeechNoLLM_MelVC.py:79-82`) to free GPU memory. Vocoding is a **separate
  offline stage** with a GTA-finetuned BigVGAN.
- **Speaker conditioning** is a 512-d L2-normalised utterance-level x-vector, loaded from
  `<target_stem>_xvector.pt`, injected via cross-attention in the conformer.
- **Duration predictor** (`use_duration_predictor=true`) predicts output length
  `N = video_len × ratio`, `ratio ∈ [dur_r_min=0.25, 1]` — output is never *slower* than
  the input. This is what makes inference length-independent of the target.
- **Two inference modes**, and the distinction matters for honest evaluation:
  - `dp` ("free", duration prediction) — the model predicts its own output length. **This
    is the real task.**
  - `tf` (teacher forcing) — the duration predictor is bypassed and the output length is
    set to the **input** length. No target leakage, but it is an easier setting.
  - Mapping: `MODE_MAP = {"tf": "tf", "dp": "free"}` in `run_eval_pipeline.py`.
- **Modality dropout** during training only (`p_av=0.85, p_video_only=0.15,
  p_audio_only=0.0`). Validation and inference are always forced to full `av`. Gated on
  `torch.is_grad_enabled()`, not `.training` (`modelSpeechNoLLM_MelVC.py:226`).
- **Mel-domain U-Net discriminator** (`mel_disc`) lives inside the model so it is
  checkpointed, but the **criterion owns its optimizer** and the warmup schedule.

### Loss (`melvc_l1`, in `criterionMelVC.py`)

Ground-truth mels are computed **on the fly**: target waveform → resample 16 k → 22.05 k →
exact BigVGAN mel. The GT mel's time dimension is fed back to the model as the
interpolation target, so predicted and GT mels are guaranteed the same length.

```
loss = mel_loss_weight * L1          (currently 0.0 — see below)
     + mse_weight     * MSE          (0.0)
     + ssim_weight    * (1 - SSIM)   (0.0)
     + gv_weight      * GV           (0.15)  global-variance, anti-oversmoothing
     + softdtw_weight * soft-DTW     (0.025, gamma=5.0)  alignment-tolerant
     + dur_loss_weight* duration     (0.1)
     + adv_weight * adv + fm_weight * feature-matching   (0.2 / 2.0, when active)
```

**Note the current recipe runs with `MEL_LOSS_WEIGHT=0.0`** — the reconstruction signal
comes from soft-DTW plus global variance plus the adversarial terms, not plain L1. Soft-DTW
gamma is the knob that was swept (`gamma5` = 5.0 is the current best).

Validation metrics computed directly on mels: `ssim_healthy`, `mcd_healthy`.

---

## 3. File map

### Model / training core (`MultiModalVC/src/`)

| File | Role |
|---|---|
| `modelSpeechNoLLM_MelVC.py` | **the model.** Config, `forward_speech`, mel head, duration predictor, discriminator attach |
| `modelSpeechNoLLM_E2E_SynthVC.py` | parent class; `_run_pipeline_to_conformer` (L203) is the shared fusion→Q-Former→upsample→conformer path |
| `criterionMelVC.py` | **the loss** (`melvc_l1`), GT mel computation, adversarial schedule, SSIM/MCD metrics |
| `dataset_pathological_finetune.py` | **the dataloader.** Manifest parsing, (input, target) expansion, x-vector loading, noise injection |
| `task_pathological_finetune.py` | fairseq task; owns the freeze/trainable logic and OGM |
| `mel_discriminator.py` | U-Net mel discriminator + `disc_loss` / `gen_adv_loss` / `feature_matching_loss` |
| `soft_dtw_cuda.py` | soft-DTW implementation |
| `bigvgan_mel.py` | BigVGAN-exact mel + resample (must match the vocoder bit-for-bit) |
| `divise_conformer/encoder_xattn.py` | conformer with interleaved content cross-attention |
| `conf/mms-speech-nollm-melvc.yaml` | base hydra config |

### Launch scripts (`MultiModalVC/scripts/`)

| File | Role |
|---|---|
| `melvc-NODEB-warp.sh` | **the training launcher.** 4×H100, 72 h. Everything is env-var driven. Supports `RESUME=true` (restores optimizer+epoch and reuses saved overrides) vs `FINETUNE_FROM_MODEL` (weights only, fresh optimizer/epoch) |
| `run_eval_pipeline.py` / `.sh` | **in-domain eval**, 5 stages. Spec (CHECKPOINTS/VOCODERS) is inline in the `.py`; `OUT_ROOT`/`MODALITY`/`MODE`/`ASR_MODEL` are env-overridable |
| `run_eval_pipeline_unseen.py` / `.sh` | same for unseen speakers; also `MANIFEST`/`TRANSCRIPTS`/`LANGUAGE` env-overridable |
| `dump_mels_eval.py` | stage 1 — checkpoint → predicted mels (`.npy`) |
| `dump_wavs_eval.py` | stage 2 — mels → wavs via a BigVGAN generator |
| `score_mos.py` | stage 3 — DNSMOS Pro + UTMOS (reference-free) |
| `score_secs.py` | stage 4 — SECS, ECAPA-TDNN cosine (`speechbrain/spkrec-ecapa-voxceleb`) |
| `score_asr.py` | stage 5 — Whisper WER/CER; `--dump-hyps` writes uid/ref/hyp |
| `score_external_wavs.py` / `.sh` | score wavs from an **external** model (e.g. QuickVC) by symlinking them into the expected layout |
| `dump_mels_gta.py`, `bigvgan_gta_finetune.sh` | GTA vocoder finetuning |

### Eval pipeline stages

1. dump mels → `OUT_ROOT/<run>/<run>-<ckpt>-npy/<uid>.npy`
2. vocode → `<npy_folder>__<vocoder_tag>/` (one folder per checkpoint × vocoder)
3. MOS → `mos_scores.txt` in each wav folder
4. SECS → `secs_scores.txt`
5. ASR → `asr_<model>.txt`, plus `asr_healthy.txt` (ceiling) / `asr_source.txt` (floor)

Logs: `MultiModalVC/exp/eval-logs/melvc_eval_pipe_<jobid>.log` (and `..._unseen_...`).

---

## 4. Data

### Manifest format — 9 tab-separated columns

```
video_path  pathological_path  file_path  synth1 synth2 synth3 synth4 synth5 synth6
```

- **col 1** `video_path` — the `.mp4`
- **col 2** `pathological_path` — **the model's audio INPUT**
- **col 3** `file_path` — **the TARGET** (ground truth); the **speaker x-vector is taken
  from this column** (`dataset_pathological_finetune.py:373`)
- **cols 4-9** `synth1..6` — alternate targets. Used only when `number_of_synths > 0`;
  **the current recipe runs `number_of_synths=0`, so they are placeholders.**

**Two parsing traps — read before editing any manifest:**

1. `load_pathological_manifest` does `line.strip().split("\t")` and then hard-requires
   exactly 9 fields (`dataset_pathological_finetune.py:150-154`). `strip()` **removes
   trailing tabs**, so a row ending in empty synth columns parses as 3 fields and crashes
   at startup. **Never leave synth columns empty** — fill them with the target path.
2. `n_avail_synths = len(target_list) - 1` is always 6 regardless of content
   (`:229`), so empty strings would be read as paths if `number_of_synths` were raised.

Line 1 of the file is a header/root line and is skipped.

**Length limit: 525 video frames @ 25 fps = 21.0 s** (`task.max_sample_size: 525`,
`min_sample_size: 10` = 0.4 s). Over-limit rows are **not rejected — they are cropped**
(`max_trim_sample_size` inherits `max_sample_size`). For reconstruction rows a crop is
merely lossy; **for conversion rows it misaligns source and target**, so oversized rows
must be removed explicitly.

There is a `<manifest>.sizes.json` cache of video frame counts, keyed on row count +
mtime. It self-invalidates, but deleting it forces a re-probe of every video (minutes).

### Datasets

| Corpus | Location | Notes |
|---|---|---|
| RECS | `VoiceConversion-fwf/RECS/` | the pathological German data |
| Sync recordings | `VoiceConversion-fwf/Sync_recordings_cropped/` | small paired set |
| YouTube | `VoiceConversion-fwf/youtubePlaylistDataset/` | German healthy |
| VoxCeleb2-German | `voxceleb2-V2/VoxCeleb2-German/` | German healthy |
| LRS3 | `lrs3-preprocessed/lrs3_video_seg24s/pretrain/` | English; 5,088 speakers, 132,959 clips, segments capped at 24 s |

**uid scheme:** `<spk>-<COND>-<utt>`, derived from parent folders
`<spk>-processed-<COND>/<folder2>`. Transcripts are keyed by
`_utt_of(uid) = uid.split("-", 2)[2]`.

---

## 5. Training stages

### Stage A — pretraining (speech reconstruction)

- Manifest: `VoiceConversion-fwf/pre-trainVoxcelebYoutube/` (111,366 train / 11,077 valid)
- Composition: 76,449 VoxCeleb2-German, 22,756 LRS3 (English), 12,160 YouTube
- Task: **reconstruction** — col2 == col3, the model rebuilds the input
- Result: `exp/multiModalVC-synth/melvc_de_warp_softdtw_adv2/checkpoints/checkpoint73.pt`
  — referred to throughout as **"ck73"** or "the pretrained checkpoint"

**Purpose:** learn audio-visual alignment and a general acoustic-phonetic representation
across many speakers and two languages. It has **no voice-conversion ability** — it was
never trained on a task where input speaker ≠ target speaker.

### Stage B — pathological finetuning

Initialised from a pretrained checkpoint; learns pathological → healthy conversion.

**The original failure.** The first finetuning manifest was ~94% pathological RECS data:
343 unique sentences repeated ~30× each. In-domain WER improved monotonically while
**unseen WER degraded monotonically (0.314 → 0.501)** — textbook catastrophic forgetting.

A memorization test on Whisper hypotheses showed the forgetting is **phonetic, not
lexical**: zero training 4-grams appeared in outputs. The model had not memorised
sentences; its decoder had collapsed onto a narrow output distribution.

Per-module drift, measured between the pretrained and finetuned checkpoints, localises it:

| module | relative L2 drift |
|---|---|
| encoders (AV-HuBERT + Whisper) | **0.075** — barely moved |
| afeat_1d_conv | **0.417** — moved most |
| conformer | 0.307 |
| duration predictor | 0.254 |
| mel_head | 0.215 |
| Q-Former | 0.212 |
| proj | 0.147 |

**The encoder representation survives finetuning. The decoder stack is what overfits.**
This is the single most useful diagnostic in the project.

**The fix — the "balanced" manifest.** Put diverse data *back into finetuning*:

`VoiceConversion-fwf/dub-autoalign-healthyYT-Trim-Balanced/trainPATH-HE.tsv`

| Source | Rows | Share |
|---|---|---|
| RECS (pathological, 50% per `<spk>-<target>` pair) | 5,099 | 51.1% |
| YouTube | 1,934 | 19.4% |
| VoxCeleb2-German | 1,000 | 10.0% |
| LRS3 healthy→healthy (English) | 846 | 8.5% |
| **LRS3 converted→original** (see below) | **856** | **8.6%** |
| Sync recordings | 243 | 2.4% |
| **total** | **9,978** | diverse share ≈ 49% |

Valid: `validPATH-HE.tsv` (1,204 rows, RECS + Sync only — **contains no LRS3**, so
validation loss is blind to the conversion rows). Test: `testPATH-HE.tsv` (1,608, all RECS).

This **flattened the forgetting curve completely**: unseen German WER stayed in
0.271–0.305 across all 16 checkpoints of the 800-epoch run, instead of climbing to 0.501.

### Stage C — the VC-capability rows (newest addition, 2026-09-13)

**Problem:** the model only ever saw 5 distinct target speakers, and pretraining taught it
no conversion at all.

**Approach:** take 1,000 LRS3 clips used in **no** existing manifest, convert each with
**QuickVC** toward one of 19 reference speakers (VCTK / LibriSpeech / LJSpeech), then build
rows where:

- col 2 (input) = **the QuickVC conversion** (a foreign voice)
- col 3 (target) = **the original LRS3 wav** (x-vectors already exist)

So the model learns *converted voice → original voice*, i.e. genuine cross-speaker voice
conversion, against **856 distinct target speakers**. QuickVC preserves input duration
exactly, so source/target frame alignment holds (verified: 0 length mismatches).

144 of the 1,000 were dropped (141 over 21 s, 3 under 0.4 s), leaving 856.

Helper manifest kept for reference:
`dub-autoalign-healthyYT-Trim-Balanced/1000newLRS3.tsv` (856 rows, already merged in).
QuickVC source map: `QuickVC-VoiceConversion/1000newLRS3-QuickVC.tsv`;
converted wavs: `QuickVC-VoiceConversion/dumpedquickvc_lrs3/` (named `<spk>_<utt>.wav`).

---

## 6. Checkpoints that matter

| Name | Path | What it is |
|---|---|---|
| **ck73** (pretrained) | `exp/.../melvc_de_warp_softdtw_adv2/checkpoints/checkpoint73.pt` | reconstruction pretraining |
| **balanced-700 / -800** | `exp/.../melvc_gamma5_balanced/checkpoints/checkpoint{700,800}.pt` | the balanced finetune |
| **C** (best merged) | `wiseft_checkpoints/balanced700_ck73/wiseft700ck73_a0.8.pt` | WiSE-FT merge, α=0.8 |
| **balanced-cont** | `exp/.../melvc_gamma5_balanced_cont/checkpoints/checkpoint{50,100,150}.pt` | continuation on the 9,978-row manifest |
| gamma5 (old) | `exp/.../melvc_gamma5/checkpoints/` | pre-balanced; best in-domain WER (0.121) but does not generalize |

Symlink trees `wiseft_bal700_ck73/checkpoints/`, `wiseft_bal700_pre/`, `modular_merge/`
point at real files in `wiseft_checkpoints/`. They exist only so the eval pipeline derives
nice run names from the parent directory. **They are symlinks, not copies.**

### WiSE-FT (weight interpolation)

```
theta_merged = alpha * theta_finetuned + (1 - alpha) * theta_pretrained
```

`alpha` = the share of the **finetuned** model. Checkpoint C is α=0.8 of balanced-700 with
0.2 of ck73 — verified by solving for α from the weights themselves (recovered 0.800000,
std 8e-7, over 828 tensors). An α sweep {0.2, 0.4, 0.5, 0.6, 0.8} was run; **0.8 won.**

### Module-wise merging — tried and FAILED

Instead of a global α, whole modules were taken from one checkpoint or the other (v1/v2/v3/v5).
**Every variant was far worse than C and worse than doing nothing** (in-domain WER 0.659–1.573
vs C's 0.209). The ranking was monotone in how much of the pretrained side got in.

**Conclusion: the modules are not independently interchangeable.** The decoder stack has
co-adapted to its own front end; handing it a foreign one breaks it. Interpolation works
precisely because it keeps every module in the same loss basin (linear mode connectivity).
**Do not revisit hard module swapping.**

---

## 7. Results

All numbers: audio-visual, duration prediction (`dp`), vocoder
`bigvganGTA/p2h_from_de_stage2_OGM_ck500/g_best_mrstft`, ASR `openai/whisper-large-v3`.
**WER corpus** = pooled edits / pooled words (no std is definable for it).

### In-domain test split (1,608 utts)

| Model | WER corpus | WER mean | CER | SECS | DNSMOS | UTMOS |
|---|---|---|---|---|---|---|
| Floor (pathological input) | 0.313 | 0.514 | 0.326 | — | 1.964 | 1.395 |
| Ceiling (healthy target) | 0.029 | 0.084 | 0.053 | — | 2.492 | 3.520 |
| balanced-700 | 0.188 | 0.245 | ~0.15 * | 0.636 | 2.679 | 3.179 |
| C (α=0.8) | 0.209 | 0.349 | 0.202 | 0.603 | **2.909** | **3.359** |
| **balanced-cont ck50** | **0.180** | 0.258 | **0.150** * | **0.646** | 2.669 | 3.170 |

### Unseen German (240 utts)

| Model | WER corpus | CER | SECS | DNSMOS | UTMOS |
|---|---|---|---|---|---|
| Floor | 0.142 | 0.067 | — | 1.818 | 1.425 |
| balanced-700 | 0.275 | 0.165 | 0.478 | 2.864 | 2.859 |
| **C** | **0.257** | **0.140** | 0.432 | **3.017** | **3.039** |
| balanced-cont ck50 | 0.276 | 0.158 | 0.471 | 2.864 | 2.868 |

### Unseen English (187 utts)

| Model | WER corpus | CER | SECS | DNSMOS | UTMOS |
|---|---|---|---|---|---|
| Floor | 0.039 | 0.029 | — | 1.488 | 1.305 |
| balanced-700 | 0.337 | 0.254 | 0.486 | 2.702 | 2.865 |
| **C** | **0.283** | **0.196** | 0.401 | **2.821** | **3.045** |
| balanced-cont ck50 | 0.306 | 0.224 | **0.494** | 2.570 | 2.803 |

### Reading the tables — four traps

1. **`*` CER outlier artifact.** Per-utterance CER = edits / reference-characters. Utterance
   `0010-BM-keywords_0089` has the reference `"du"` — **2 characters**. A hallucinated
   Whisper output gives CER = 221.5 for that one utterance, which drags ck50's reported CER
   mean from 0.150 to 0.288 and balanced-700's to 0.349. **Always check for short-reference
   outliers before trusting a CER mean.** Corpus WER is pooled and therefore robust.
2. **The unseen "ceiling" is meaningless.** Its WER is >1.0 because those healthy reference
   recordings do not say what the transcripts say. Use the **floor** as the reference there.
3. **DNSMOS above the healthy ceiling** (C scores 2.909 vs the real recordings' 2.492) is a
   known bias of reference-free MOS predictors toward clean, slightly over-smoothed vocoder
   output. It does **not** mean the output beats real speech.
4. **Merging raises DNSMOS/UTMOS.** C's quality advantage over both balanced-700 and ck50
   comes from the merge with the pretrained checkpoint, not from anything the finetune lost.

### Baselines

**QuickVC** (external VITS-based VC, `QuickVC-VoiceConversion/`) was scored on all three
sets via `score_external_wavs.py`. It **degrades** the input everywhere (in-domain WER 1.360
vs a floor of 0.314). Its apparent UTMOS advantage on unseen sets is a **length confound** —
it copies input duration exactly (ratio 1.000). It uses **no perceptual loss**; its losses
are `loss_gen + loss_fm + loss_mel + loss_kl + loss_subband`.

---

## 8. Current state (2026-09-14)

**Running:** `melvc_gamma5_balanced_cont`, SLURM job 1728956, ~5 min 17 s/epoch, at epoch
~175 of 800. Initialised from balanced-800 via `FINETUNE_FROM_MODEL` (weights only → epoch
counter restarts at 1). Discriminator forced active from update 0.

Launch command used:

```
SOFTDTW_GAMMA=5.0 RUN_NAME=melvc_gamma5_balanced_cont WANDB_NAME=melvc_gamma5_balanced_cont \
WANDB_RUN_ID=melvc_gamma5_balanced_cont MAX_EPOCH=800 SAVE_INTERVAL=50 KEEP_LAST_EPOCHS=-1 \
WANDB_PROJECT=melvc \
FINETUNE_FROM_MODEL=.../melvc_gamma5_balanced/checkpoints/checkpoint800.pt \
FINETUNE_DATA=.../dub-autoalign-healthyYT-Trim-Balanced \
NOISE_PROB=0.5 P_AV=0.85 P_VIDEO_ONLY=0.15 P_AUDIO_ONLY=0.0 MODALITY_MODE=av \
OGM_ENABLED=true OGM_ALPHA=0.03 WHISPER_TOP_N=2 WHISPER_LN_TRAINABLE=true \
AVHUBERT_TOP_N=2 AVHUBERT_LN_TRAINABLE=true USE_CONTENT_XATTN=true \
CONTENT_INTERLEAVE_START=odd USE_DUR_PRED=true DUR_PRED_LAYERS=2 DUR_R_MIN=0.25 \
SOFTDTW_WEIGHT=0.025 DUR_LOSS_WEIGHT=0.1 MEL_LOSS_WEIGHT=0.0 SSIM_WEIGHT=0.0 \
MSE_WEIGHT=0.0 GV_WEIGHT=0.15 TRANSCONV_LAYERS=2 QUERIES_PER_SEC=4 MEL_BANDS=80 \
LR=1e-5 WEIGHT_DECAY=0.05 UPDATE_FREQ=1 WARMUP_UPDATES=500 MAX_UPDATE=3000000 \
BATCH_SIZE=24 NUM_WORKERS=8 VALIDATE_INTERVAL=1 VALIDATE_INTERVAL_UPDATES=0 \
USE_DISCRIMINATOR=true FORCE_DISC_ACTIVE=true DISC_PRETRAIN=false DISC_START_UPDATES=0 \
ADV_WARMUP_UPDATES=500 ADV_WEIGHT=0.2 FM_WEIGHT=2.0 DISC_LR=1e-5 DISC_BETA1=0.8 \
DISC_BETA2=0.99 DISC_GRAD_CLIP=20.0 FREEZE_DISC=false sbatch scripts/melvc-NODEB-warp.sh
```

**Training validation loss is flat (~1.57-1.60) and this is expected.** The valid manifest
contains only RECS + Sync — **no LRS3 conversion rows** — so it cannot see the capability
being added. Judge this run by the eval pipeline, never by `valid_loss`.

**Where ck50 landed:** best in-domain WER (0.180) and best in-domain SECS (0.646) of any
generalizing checkpoint, and the best unseen-English SECS overall (0.494). But it has **not**
matched C's unseen WER. The obvious next step is to **merge a continuation checkpoint with
ck73 at α=0.8** and check whether it inherits both advantages.

---

## 9. Working practices and hard-won gotchas

### Environment

- Conda env `torchEnv`; python at `/home/eo49197/.conda/envs/torchEnv/bin/python`.
- **`ffmpeg` / `ffprobe` exist in the env's `bin` but are NOT on `PATH` inside SLURM jobs.**
  Any script shelling out to them must prepend the env bin. This has caused two separate
  multi-hour failures. Prefer `cv2` (already a dependency) for video frame counts and
  `soundfile` for audio metadata.
- **Do not install packages.** Report the needed dependency to the user instead.
- **Never launch GPU jobs on the user's behalf.** Provide the `sbatch` command; the user runs it.

### Evaluation

- Corpus WER and per-utterance mean WER are **different metrics**. Report which one.
- Always include the **floor** (pathological input) and **ceiling** (healthy target) rows.
- The floor/ceiling values are stable across runs — use them to confirm two evaluations are
  comparable before comparing model rows.
- Re-scoring an existing checkpoint is wasted compute; results persist in
  `DumpedMels_*/<run>/<...>__<vocoder>/{asr_*,secs_scores,mos_scores}.txt`.

### Denoising utility (incomplete)

`scripts/denoise_targets.py` denoises manifest targets in place with a `_denoise` suffix.
Two bugs were found and one is **still unfixed**:

- FIXED: the stem selector matched the bare substring `"vocals"`, which also matches the
  **model filename** `vocals_mel_band_roformer` — so it kept the *noise* stem and produced
  672 near-silent files. It now matches the parenthesised label `(vocals)` only.
  (`_(vocals)_` = SPEECH, `_(other)_` = noise. Measured energy ratio 0.961 vs 0.004.)
- **NOT FIXED:** inputs shorter than ~2.55 s crash audio-separator with
  `The size of tensor a (0) must match the size of tensor b (66150)`. 936 of 1,608 files
  failed this way. **The verified fix — pad to 6 s, separate, trim back — has been tested
  but not applied.**

### A note on prior mistakes worth not repeating

- Blanket statements like "X is best" are usually wrong here, because the ranking genuinely
  differs by metric and by domain. C wins unseen WER; ck50 wins in-domain WER and SECS;
  gamma5-old wins raw in-domain WER but does not generalize. Say which metric, which set.
- Do not add checkpoints, baselines, backups, or files that were not asked for.
