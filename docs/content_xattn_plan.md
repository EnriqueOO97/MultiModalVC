# Content Cross-Attention Warp — Implementation Plan (draft)

Status: **plan only, no code changes.** Scope: let the conformer learn the
non-uniform pathological→healthy time warp so we can train on healthy targets
*as-is* (drop the offline DTW time-stretch and its artifacts).

---

## 0. The idea in one paragraph

Today the conformer does pure self-attention on the **post-upsample acoustic
frames** (Q=K=V, length N). It never looks back at the compressed Q-Former
tokens. We add, per conformer block, a **content cross-attention** sub-layer:

- **Q** = the N acoustic frames (running decoder state, as now)
- **K/V** = the Q-Former "content tokens" (pre-upsample, length ≈ `len_queries`)

Each of the N output slots pulls the content it needs → a **learned
non-uniform warp**. The content tokens are derived from the input, so they
**exist at inference** — no target-availability problem. This is inference-safe
and orthogonal to the duration predictor (which still sets N).

Why this may work where the single-step cross-attn upsampler failed: 12 layers
of refinement instead of one, and Q is initialised from the uniform-upsample
(content already in roughly the right monotonic order) so cross-attn only
learns the *residual* warp.

### Two tasks, do not conflate

- **Task 1 — produce N** (how many mel frames the output has). Solved by the
  **duration predictor** (§8). One scalar per utterance.
- **Task 2 — the warp** (which content lands on which of those N frames).
  Solved by the **content cross-attention** (§0–§2) + loss safeguards (§4).

They are independent modules. You need one of each. The "3 options"
(soft-DTW / autoregressive decoder / cross-attn upsampler) were all Task 2;
content cross-attn is the chosen Task-2 mechanism. The duration predictor is
Task 1 and is orthogonal to all of them.

---

## 1. Files touched (smallest diff)

Reuse the existing cross-attn machinery — the layer already has a `cross_attn`
sub-layer for `spk_emb`. We copy that pattern for a second memory. Four files:

| File | Change |
|------|--------|
| `src/divise_conformer/encoder_layer_xattn.py` | add one optional `content_attn` sub-layer + `content_mem`/`content_mask` kwargs in `forward` |
| `src/divise_conformer/encoder_xattn.py` | build the `content_attn` module; thread `content_mem`/`content_mask` through `MultiSequentialWithCrossAttn` and both `forward`s; add `content_layers` gate |
| `src/modelSpeechNoLLM_E2E_SynthVC.py` | in `_run_pipeline_to_conformer`: keep Q-Former tokens as `content_mem`, project 1024→512, pass tokens+mask into `self.conformer(...)` |
| `src/conf/mms-speech-nollm-melvc.yaml` | flags: `use_content_xattn`, `content_layers`, `guided_attn_weight` |

No new module classes, no new files. `MultiHeadedAttention` (already imported)
is the content-attn; the plumbing mirrors `spk_emb`.

---

## 2. Hook points (concrete)

### 2a. Keep the content memory (`modelSpeechNoLLM_E2E_SynthVC._run_pipeline_to_conformer`)

The Q-Former output is `query_output` (→ `queries`, dim `qformer_dim=1024`,
length `len_queries`) at [L263-267]. Today it's consumed by `proj1` + upsample
and discarded. Instead:

```
# after query_output is built, before upsampling
content_mem  = self.content_proj(query_output)          # 1024 -> 512, new nn.Linear
content_mask = build_mask(query_lengths, max_av_len)     # reuse query_attn_mask logic
```

`content_mask` already exists in spirit — `compression_using_qformer` builds
`query_attn_mask` (B, max_len). Return it or rebuild from `len_queries`.

At the conformer call (currently `x = self.conformer(x, spk_emb=spk_emb)`,
[L331]):

```
x = self.conformer(x, spk_emb=spk_emb,
                   content_mem=content_mem, content_mask=content_mask)
```

`content_proj` is one `nn.Linear(qformer_dim, 512)` built in `__init__`, gated
by `use_content_xattn`.

### 2b. Interleaved main-attention swap (`EncoderLayerWithCrossAttn.forward`)

**Do NOT add a third attention.** Each block keeps exactly one "main" attention
+ the existing speaker cross-attn. We **swap the main attention's type** by
block parity — no size increase:

- **Odd blocks (1,3,5,7,9,11):** main attn = **content cross-attn** (replaces
  self-attn). Q = the N acoustic frames, K/V = content memory.
- **Even blocks (2,4,6,8,10,12):** main attn = **self-attn** (as today).
- **Speaker cross-attn stays in ALL 12, unchanged.**

So per block: `[main-attn (self OR content-xattn)] + speaker-xattn + conv + FFN`.
6 content-xattn + 6 self-attn, interleaved. Consequence (not a problem): odd
blocks lose self-attn *in that layer*, but even blocks still provide frame-to-
frame self-attn every other layer.

```
# main-attn slot — pick by the flag set at build time on THIS layer:
if self.main_is_content and content_mem is not None:
    x_att = self.content_attn(x_q, content_mem, content_mem, mask=content_mask)
else:
    x_att = self.self_attn(x_q, x, x, ...)   # existing self-attn path
# ... residual/norm exactly as the current self-attn wrapping ...
# speaker cross-attn (unchanged) ...
```

Each layer holds **either** `self_attn` **or** `content_attn`, decided at build
time (§2c), so no extra module lives in the block. Stash the content-attn
weights on `self._last_content_attn` (like `_last_residual_ratio`) if
`guided_attn_weight > 0` (§4), no signature churn.

### 2c. Build + gate (`EncoderWithCrossAttn.__init__`, `ConformerEncoderWithCrossAttn`)

- For each block index `i`, if `use_content_xattn and (i % 2 == 0)` build a
  `content_attn = MultiHeadedAttention(...)` and set `main_is_content=True`
  **instead of** the self-attn (block 1 = index 0 = content-xattn = "odd" in
  1-based). Even blocks keep self-attn as built today.
- `content_interleave_start` flag (default `odd`) lets you flip which parity
  gets content-xattn without code changes.
- Speaker cross-attn: built in all 12 exactly as now. **Untouched.**
- Thread `content_mem`/`content_mask` through `MultiSequentialWithCrossAttn`
  (kwarg passthrough) and both `forward` signatures.

---

## 3. Config flags (all default OFF → current behaviour unchanged)

```yaml
model:
  use_content_xattn: false        # master switch (interleaved main-attn swap)
  content_interleave_start: odd   # which block parity gets content-xattn
criterion:
  softdtw_weight: 0.0             # ADDED mel loss (soft-DTW), not a replacement (§4b)
  softdtw_gamma: 0.1              # soft-min temperature
  mel_l1_weight: 1.0              # keep frame-wise mel L1; down-weight when softdtw on
  guided_attn_weight: 0.0         # OPTIONAL diagonal prior on content-attn (§4a)
```

`use_content_xattn: false` must reproduce today's graph exactly (no content_proj,
every block keeps its self-attn, no extra loss). Reviewable and A/B-able.

---

## 4. Losses — what stays, what's added

**The mel reconstruction loss stays. Nothing is replaced.** Today's loss is a
**single-resolution frame-wise mel L1** (not multi-resolution — that's the
waveform criterion). We *keep* it and optionally *add* soft-DTW as a second,
alignment-tolerant **mel** loss (§4b). Frame-wise L1 scores the prediction under
the *identity* alignment (pred frame i vs GT frame i); soft-DTW scores it under
the *best monotonic* alignment (pred frame i vs GT frame j≈i), so it forgives a
correct-but-time-shifted phoneme. Precedent for using both / or soft-DTW as the
primary mel loss: Parallel Tacotron 2 (arXiv 2103.14574).

Caveat: a *strong* frame-wise L1 partially fights soft-DTW (L1 insists on the
identity alignment). If both are on, **down-weight L1** (`mel_l1_weight`) so it
stabilises without overriding the warp tolerance.

Separately, a non-autoregressive content cross-attn can **smear** while learning
alignment; §4a is an *optional* stabiliser for that — unrelated to soft-DTW.

### 4a. Guided / monotonic attention regularizer  *(OPTIONAL — only if the content-attn smears)*

**Concept.** In speech, input content and output frames are (almost) in the
same left-to-right order — the alignment matrix should sit near the diagonal.
"Guided attention" (Tachibana 2018) just *nudges* it there with a soft penalty;
it does not hard-constrain. It stops the attention from collapsing to a blur or
jumping around while it learns.

**Technical.** For a content-attn weight matrix `A` (B, N_out, M_content), build
a fixed cost mask `W[n,m] = 1 - exp(-((n/N - m/M)^2) / (2 g^2))` (g≈0.2). Cost
is large far from the diagonal, ~0 on it. Penalty:

```
L_guided = mean(A * W)            # elementwise, masked over valid n,m
loss += guided_attn_weight * L_guided     # weight ~0.1, optionally decayed
```

Cheap (one broadcast mul), no new deps. `W` is cached per (N,M) shape. Because
Q is initialised from the uniform upsample, A already starts near-diagonal —
this mostly *keeps* it there while the residual warp is learned.

Cost: ~15 lines + threading `A` out of the layer. **Start here.**

### 4b. Soft-DTW — the PRIMARY mel loss; frame-wise L1 becomes a small stabilizer

Both are mel-domain losses vs the same GT (the healthy parallel utterance's mel).
The "warp" is the alignment path used *in the loss* between pred and GT — nothing
is done to the GT itself.

**Concept.** Plain L1 compares pred frame *i* to GT frame *i* (identity
alignment) — it punishes a correct phoneme that's a few frames early/late.
**Soft-DTW** (Cuturi & Blondel 2017) instead finds the best *monotonic alignment*
between the two sequences and scores the aligned pairs, so it forgives internal
timing differences. It's the DTW you run offline on the *waveform*, moved into
the *loss* on the *mel* — which removes the reason you needed the preprocessing
(and its artifacts). It also tolerates pred/GT length mismatch, so N need only be
approximately right.

**Weighting (from the literature).** Soft-DTW's per-frame cost *is* L1
(`D[i,j]=‖P_i−G_j‖₁`), so on the diagonal soft-DTW ≈ frame-wise L1 — a separate
L1 is partly redundant. Parallel Tacotron 2 (arXiv 2103.14574) uses **soft-DTW as
the sole spectrogram loss** (banded, width 60), no separate L1. Since we
teacher-force N=GT-length, we *can* keep a small L1: soft-DTW primary (~1.0), L1
small auxiliary (~0.1–0.3), because L1 drives fast early sharpness/alignment
(Tacotron-2 #336) while soft-DTW can be blurry early.

**Ablation sweep (recommended order):**

| Run | `softdtw_weight` | `mel_l1_weight` | meaning |
|-----|------|------|---------|
| control | 0.0 | 1.0 | today's loss (identity-aligned L1 only) — baseline |
| exp A | 1.0 | 0.1–0.3 | soft-DTW primary + small sharpening L1 |
| exp B | 1.0 | 0.0 | pure soft-DTW (L1 still present *inside* it, on the warp path) |

Note: "mel_l1_weight=0" does NOT remove L1 — soft-DTW's internal cost is L1,
just applied along the best warp path instead of the diagonal.

**Technical.** Given pred `P` (N, 80) and GT `G` (M, 80): pairwise cost
`D[i,j] = ||P_i - G_j||` (or L1), then the DTW recursion with a **soft-min**
(log-sum-exp with temperature γ) instead of hard min, so it's differentiable:

```
R[i,j] = D[i,j] + softmin_γ(R[i-1,j], R[i,j-1], R[i-1,j-1])
L_sdtw = R[N,M]     # backprop through the soft recursion
```

Cost/ceiling: **O(N·M) per sample**, sequential recursion — heavier and not
trivially vectorised. Options, laziest first: (1) an existing impl
(`pytorch-softdtw-cuda`) — a dependency, but don't hand-roll a CUDA kernel;
(2) a small banded pure-torch version (restrict |i·M/N − j| ≤ band) to cut cost
and enforce near-monotonicity. **Do not build this until 4a is shown
insufficient** — it's the expensive rung.

### Combining

4a + a normal L1 is the minimal viable setup. If residual misalignment persists,
switch the reconstruction term to 4b (keep 4a as a light prior). They're
complementary: cross-attn supplies warp *capacity*, the loss supplies warp
*tolerance*.

---

## 5. Out of scope (dependencies, not built here)

- **Speaker cross-attn pruning** (12→fewer). Tune after content-attn lands.
- **Dropping the offline DTW preprocessing.** The payoff, but flip it only once
  4a/4b hold on a val set — don't co-change data + architecture in one run.

(Duration predictor is now IN scope — see §8.)

---

## 7. Retrocompatibility + launch script

**Rule: every new param is gated behind a flag defaulting OFF.** With flags
absent, the model builds byte-identically to today → current script works, old
checkpoints load, `RESUME=true` resumes in old mode (it replays a saved
`overrides.yaml` that has none of the new flags).

Flags ride the same `+model.*` / `+criterion.*` override channel already used in
[`scripts/melvc-NODEB.sh`](../scripts/melvc-NODEB.sh). To run the **new**
experiment, add this block to a *copy* of that script (the `fairseq-hydra-train`
invocation, alongside the existing `+model.transconv_layers=...` lines):

```bash
    +model.use_content_xattn=${USE_CONTENT_XATTN:-false} \
    +model.content_layers=${CONTENT_LAYERS:-6} \
    +model.use_duration_predictor=${USE_DUR_PRED:-false} \
    +model.dur_pred_layers=${DUR_PRED_LAYERS:-2} \
    +criterion.guided_attn_weight=${GUIDED_ATTN_WEIGHT:-0.0} \
    +criterion.dur_loss_weight=${DUR_LOSS_WEIGHT:-0.0} \
```

Poor results → drop the flags (or use the original script) → exactly back to
today. No code rollback needed.

**Warm-start vs from-scratch.** New params (`content_proj`, `content_attn`,
duration predictor) have no prior weights. Either:
- **from scratch**, or
- **warm-start the backbone** via `checkpoint.finetune_from_model=<best MelVC ckpt>`
  and init only the new params fresh. This needs a **non-strict load** (the old
  ckpt lacks the new keys) — add a `strict=False`-style path or filter missing
  keys at load. `use_content_xattn=false` runs need no such handling (keys match).

---

## 8. Duration predictor (Task 1 — produce N)

**Goal.** Predict N = number of output mel frames, per utterance, from
inference-available inputs. Replaces the current inference-time fallback
(`audio_lengths → frame formula`, [modelSpeechNoLLM_MelVC.py L163-172]); training
still teacher-forces the mel path with GT N and trains the predictor with its
own loss.

**Inputs (all available at train + inference):**
- Q-Former **content tokens** (variable length ≈ `len_queries`) — the content.
- Target **speaker embedding** (B, 512) — per-speaker rate bias.
- **input length** scalar (use `len_queries` or `video_lengths`) — see reparam.

**Architecture — mirror `Speech_Rate_Predictor`** ([sub_model/modules.py L108]):
prepend a learned `[DUR]` token to the content tokens, 1–2 small transformer
layers, take the `[DUR]` output; add the projected speaker embedding to it;
`Linear(d→1)`.

```
dur_in   = proj(content_tokens)              # to dur dim (e.g. 256)
h        = transformer([DUR] ++ dur_in)      # 1-2 layers, masked
z        = h[:,0] + proj_spk(spk_emb)        # inject speaker
log_ratio = Linear(z, 1)                     # predict a RATIO, not absolute N
N_pred   = round( input_len * exp(log_ratio) )
```

**Predict a ratio, not absolute frames.** `N = input_len · exp(log_ratio)` turns
the task into "how much faster is the healthy target than this input" — far
better-conditioned than regressing absolute frames from pooled content. (Healthy
usually faster → ratio < 1.)

**Loss (separate term, log domain):**
```
L_dur = | log_ratio_pred - log(N_gt / input_len) |     # L1 in log-ratio space
loss += dur_loss_weight * L_dur                        # ~0.1
```

**Train vs inference:**
- **Train:** mel path uses **GT N** (teacher forcing, as today) so the
  reconstruction loss stays well-posed; predictor learns alongside via `L_dur`.
- **Inference:** set `target_lengths = N_pred`. No GT needed.

**Hook:** in `_run_pipeline_to_conformer` (or just before it), compute `N_pred`
from the Q-Former tokens; when `model.training` use GT `target_lengths`, else use
`N_pred`. Stash `log_ratio_pred` on the model for the criterion to read (same
pattern as `_last_residual_ratio`).

**Check to leave behind:** `demo()` asserting (a) `exp(log_ratio)=1` gives
`N_pred=input_len`, and (b) `L_dur=0` when `log_ratio_pred = log(N_gt/input_len)`.

skipped: per-token/FastSpeech2 durations (conv stack). add when a single global
ratio proves too coarse (e.g. pauses need per-region duration, not global rate).

---

## 6. Ponytail ledger

- `use_content_xattn=false` reproduces the current graph → safe default,
  reviewable A/B. skipped: any new module class; reused `MultiHeadedAttention`
  + the `spk_emb` pattern. add when a genuinely different attention is needed.
- content-attn in **first 6 layers only**. skipped: all-12. add when val shows
  alignment unresolved in upper blocks.
- **guided-attn before soft-DTW**. skipped: soft-DTW (O(N·M), dep or custom).
  add when guided-attn alone leaves audible timing errors.
- One runnable check to leave behind when coding 4a: `demo()` asserting the
  guided cost `W` is ~0 on the diagonal and large off it, and that
  `L_guided(diagonal A) < L_guided(uniform A)`.
