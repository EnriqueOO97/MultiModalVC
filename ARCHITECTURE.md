# MultiModal Voice Conversion — Architecture Description

example of a **3-second audio-visual clip**: audio sampled at 16 kHz (48,000 samples), video at 25 fps (75 frames).

---

## Overview

pipeline:

```
Video  ──► AV-HuBERT ───┐
                        ├──► Fusion ──► Q-Former ──► proj1 ──► Interpolation ──► proj2 ──► Conformer ──► HiFi-GAN ──► Waveform
Audio  ──► Whisper ─────┘                                                                     ▲
                                                                                      Speaker Embedding
```

The speech rate predictor (a side branch off the raw Whisper features) determines the number of Q-Former queries before the main pipeline starts.

---

## 1. Video Encoder — AV-HuBERT (Large)

**Input (3s example):**

**Output:**
- `(B, 75, 1024)` — one 1024-dimensional feature vector per video frame, at 25 Hz
- **This module is frozen** throughout all training phases. No gradients flow through it.

**Key point:** AV-HuBERT encodes the visual articulatory dynamics (lip movements) at exactly 25 Hz, one token per frame. The output sequence length is therefore directly determined by the number of video frames.

---

## 2. Audio Encoder — Whisper Medium (Encoder only)


Whisper by default produces 1500 frames for a 30-second window. Regardless of the input length. Therefore, only the relevant information (3s) has to be extracted 

**Input (3s example):**
- Whisper always pads the waveform to exactly **30 seconds** (480,000 samples) before encoding.
- A 3-second clip is therefore zero-padded to 30 seconds internally.

**Raw output:**
- `(B, 1500, 1024)` — always 1500 frames at 50 Hz regardless of actual content duration.
  The first 150 frames correspond to the actual 3 seconds of speech; the remaining 1350 correspond to silence/padding.

**This module is frozen** throughout all training phases.

### Temporal alignment with AV-HuBERT (afeat_1d_conv)

To align Whisper features with the 25 Hz video rate, a learned 1-D convolution (`afeat_1d_conv`) downsamples by 2×:
```
Conv1d(in=1024, out=1024, kernel=2, stride=2)
(B, 1500, 1024)  ──►  (B, 750, 1024)  
    [50 Hz        →         25 Hz]
```

The result is then **trimmed** to match the AV-HuBERT sequence length `T_v`:
```
whisper_proc = whisper_proc[:, :T_v, :]
(B, 750, 1024)  ──►  (B, 75, 1024)
```

After this step both encoders have produced `(B, 75, 1024)` at 25 Hz, perfectly aligned frame-by-frame with the video.
---

## 3. Speech Rate Predictor

**Purpose:** Before anything is fused or compressed, the model needs to decide how many Q-Former query tokens to allocate for this particular utterance. That number should depend on how much phonetic content there is. The speech rate predictor estimates the speaking rate and produces a **multiplicative scaling factor** that directly controls the number of Q-Former queries.

**This module is frozen** throughout all training phases (pretrained weights are loaded at model creation, and no gradient flows through it).
---

**Input:** The raw Whisper encoder output (1500 frames at 50 Hz, before any downsampling), restricted to the portion that corresponds to the actual audio content:

```python
whisper_enc_out[:, :2*T_v, :][:, ::4, :]
```

Step by step for the 3-second example:
1. `whisper_enc_out` is `(B, 1500, 1024)` — the full 30-second padded Whisper output
2. `[:, :2*T_v, :]` = `[:, :150, :]` — keep only the first 150 frames. The factor of 2 comes from the rate mismatch: Whisper runs at 50 Hz and the video runs at 25 Hz, so for every video frame there are 2 Whisper frames. Therefore `2 × T_v = 2 × 75 = 150` Whisper frames = 3 actual seconds of audio.
3. `[:, ::4, :]` — take every 4th frame of those 150, giving ~38 frames. This is a computational efficiency trick: the speech rate predictor only needs to estimate an overall speaking rate, not fine-grained temporal detail, so a coarse 12.5 Hz sampling is sufficient.

Final input to the predictor: `(B, 38, 1024)`
---

**Output: a scalar speech rate factor, clamped to the range [1.0, 2.0].** A value of 1.0 means normal speaking rate. A value of 2.0 means the speaker is talking twice as fast as the baseline.
---

**How the factor is used:**

The factor is passed forward and used in the Q-Former stage (described in Section 5). At this point, the only thing the speech rate predictor does is produce that scalar. Its downstream effect is explained once the Q-Former has been introduced.

The calculation that converts the factor into a query count is:
```python
# T_v / 25 converts frame count to seconds  (25 = video fps, not queries_per_sec)
base_queries     = T_v / 25 * queries_per_sec   # duration_in_seconds × queries_per_sec
                                                 # = 3s × 3 = 9 base queries
adjusted_queries = int(base_queries * factor)    # scale by speech rate
N_queries        = max(adjusted_queries, queries_per_sec)  # enforce minimum of 3
```

For the 3-second example:
- Normal speech (factor = 1.0): `N_queries = max(9, 3) = 9`
- Fast speech (factor = 1.5): `N_queries = max(13, 3) = 13`
- Very fast speech (factor = 2.0): `N_queries = max(18, 3) = 18`

`N_queries` and the factor itself are both passed to the Q-Former. What they do there is explained in Section 5.

---

## 4. Modality Fusion

Both modalities are now aligned at `(B, 75, 1024)`. They are combined into a single audio-visual feature tensor. Three strategies are supported:

| Strategy | Operation | Output shape |
|---|---|---|
| `concat` | Concatenate along feature axis | `(B, 75, 2048)` |
| `add` | Element-wise addition | `(B, 75, 1024)` |
| `cross-att` | Cross-attention (visual queries attend to audio keys/values) | `(B, 75, 1024)` | (im doing cross attention for the fusion because they gave better results than the other methods, but it is computationally more expensive too)

**Modality dropout (training only):** With configurable probability the audio or video stream is replaced by a learned mask embedding (`audio_mask_emb` or `video_mask_emb`, each 1024-dimensional) before fusion. This forces the model to remain functional when one modality is unavailable, improving robustness to missing inputs at inference time.

## 5. Q-Former (Query Transformer)

It compresses the fused audio-visual feature sequence — which at this point is `(B, 75, fused_dim)` — into a small, fixed set of compact tokens. Those tokens are what the rest of the pipeline (projections, conformer, vocoder) will work with. an the number of output tokens is defined by N_queries, in the speech rate predictor formula

Before running the cross-attention, the Q-Former first rescales the fused AV feature sequence using the speech rate factor produced earlier. Concretely, for each sample in the batch:

```python
resized_av_feat = F.interpolate(av_feat, size=int(factor * T_v), mode='linear')
```

For the 3-second example:
- factor = 1.0 → AV sequence stays at 75 frames
- factor = 1.5 → AV sequence is stretched to 112 frames
- factor = 2.0 → AV sequence is stretched to 150 frames

The rationale: the `N_queries` count was already scaled up by the same factor (more queries for faster speech). By also stretching the AV feature sequence proportionally, the Q-Former sees more key-value pairs to attend to — the density of phonetic information per frame stays roughly constant regardless of speaking rate. Without this rescaling, a fast speaker would have the same 75 AV frames being compressed into 18 queries, and the queries would have far less to attend to per token.


**Output:**
```
(B, N_queries, 1024)   [e.g. (B, 9, 1024) for a 3s clip at normal speech rate]
```

The Q-Former is the main information bottleneck of the model. It distils a variable-length audio-visual sequence into a small, fixed-format set of tokens that summarises the linguistic content.


## 6. Projection 1 + Layer Norm (proj1 / ln1)

```
Linear(1024 → 768)  +  LayerNorm(768)
(B, N_queries, 1024)  ──►  (B, N_queries, 768)
```

## 7. Temporal Upsampling (Interpolation)

The Q-Former output has `N_queries` tokens (coarse-grained, ~25 Hz equivalent). The vocoder requires one feature frame per mel spectrogram frame. The target number of mel frames is derived from the audio length:

```
target_frames = floor((audio_samples + 2 * pad − n_fft) / hop_length) + 1
```

With `n_fft = 1024`, `hop_length = 160`, `pad = (n_fft − hop_length) / 2 = 432`:

```
for a 3s audio at 16kHz, and a vocoder with an upsampling factor of 160x
the waveform would have 48000 samples, then the input that the vocoder needs to create such waveform is computed like this

target_frames = floor((48000 + 864 − 1024) / 160) + 1
              = floor(47840 / 160) + 1
              = 299 + 1
              = 300
```

**For the 3-second example: 300 mel frames.**

Each sample in the batch is independently linearly interpolated from its valid `N_queries` length to its `target_frames` length:

```
(B, N_queries, 768)  ──► linear interpolate ──►  (B, 300, 768)
```

This step bridges the compressed linguistic representation back to the acoustic time resolution required for speech synthesis.

---

## 8. Projection 2 + Layer Norm (proj2 / ln2)

```
Linear(768 → 512)  +  LayerNorm(512)
(B, 300, 768)  ──►  (B, 300, 512)
```

---
Before this input goes into vocoder, first it is refined by the conformer, because after interpolation the information is perturbed by this mechanic step.
## 9. Conformer with Speaker Cross-Attention

**Standard Conformer blocks** interleave:
- Feed-forward sub-layer
- Multi-head self-attention (captures long-range linguistic dependencies)
- Convolution sub-layer (captures local phone-level dynamics)
- Second feed-forward sub-layer
- Layer normalisation and residual connections at each stage

**Cross-attention to speaker embedding:**  
The target speaker identity is provided as a 512-dimensional embedding (precomputed, supplied via the dataset manifests). It is reshaped to `(B, 1, 512)` and injected into every Conformer layer via cross-attention:

```
x_out = x + CrossAttention(query=x, key=spk_emb, value=spk_emb)
```

This is the mechanism by which the model continuously imprints the target voice identity onto the latent representation. Because the speaker embedding is attended to at every layer, the voice characteristics are not added superficially but are woven into the feature representation throughout the Conformer's depth.

The Conformer does not change the temporal dimension:

```
Input:  (B, 300, 512)  +  spk_emb (B, 1, 512)
Output: (B, 300, 512)   ← these are the "latent features"
```

Followed by a final layer norm (`ln3`). The output is referred to as:

## 10. Vocoder — HiFi-GAN

**What it is:** A neural vocoder that generates a raw audio waveform directly from the conformer's 512-dimensional latent features. It replaces the traditional mel spectrogram intermediate representation.

**Adapter (`vocoder_conv_pre`):**  
A weight-normalised `Conv1d(512, 512, kernel=7, padding=3)` that adapts the conformer output distribution to what the HiFi-GAN upsampling chain expects. The original HiFi-GAN was designed to receive 128-bin mel spectrograms; this adapter substitutes that entry point.

**Upsampling chain:**  
Four transposed-convolution stages progressively upsample the time axis. Their combined stride equals the STFT hop length:

```
10 × 4 × 2 × 2 = 160   (matches hop_length = 160)
```

Each upsampling stage is followed by multi-receptive-field fusion (MRF): three residual blocks operating at different kernel sizes in parallel, whose outputs are averaged.

**Final layers:** LeakyReLU → `Conv1d(→1)` → `tanh` producing a single-channel waveform in `[-1, 1]`.

```
Input:  (B, 512, 300)   ← conformer output transposed
Output: (B, 1, 48000)   ← 3 seconds at 16 kHz
```

 `300 frames × 160 samples/frame = 48,000 samples = 3.0 s at 16 kHz` ✓

---

## 11. Discriminators (Phase 2 only)

Two complementary discriminators are used during adversarial training to provide perceptual realism losses.

**Multi-Period Discriminator (MPD):** Operates on different periodic sub-samples of the waveform (periods 2, 3, 5, 7, 11). Each sub-discriminator reshapes the waveform into a 2-D grid and runs 2-D convolutions, making it sensitive to periodic structures inherent in voiced speech.

**Multi-Scale STFT Discriminator (MS-STFTD):** Operates in the frequency domain at five STFT resolutions (`n_fft` ∈ {512, 1024, 2048, 4096, 8192}). Captures spectral realism across a wide range of frequencies and time scales.

Both discriminators are excluded from Fairseq's main parameter optimiser and updated by a separate discriminator optimiser managed within the criterion. They are only active during Phase 2.

---

## Training Strategy

### Phase 1 — Speech Reconstruction (no discriminator)

The forward pass performs **two internal steps** per batch, using the same video and the same target speaker embedding.

**Step 1 — Canonical pass:**
1. Input: canonical video + **canonical audio**, speaker embedding of the canonical speaker
2. Full pipeline → conformer → vocoder → waveform
3. The conformer output is saved as the **teacher (canonical) latent features**
4. Loss: **Mel loss** — L1 distance between the predicted mel spectrogram (derived from the generated waveform) and the ground-truth canonical mel spectrogram

The model is not doing voice conversion here. It is doing speech reconstruction in an autoencoder fashion: canonical audio in, canonical audio out. This teaches the full pipeline to produce intelligible, natural-sounding speech.

**Step 2 — Synthetic pass:**
1. Input: canonical video + **synthetic audio** (same utterance, different voice identity), same canonical speaker embedding
2. Pipeline runs only up to the conformer — the vocoder is **not called**
3. The conformer output is the **target (synthetic) latent features**
4. Loss: **Conversion loss** — L1 distance between the synthetic latent features and the canonical latent features saved in step 1

The synthetic audio is a voice-converted version of the canonical audio: it carries the same linguistic content but a different voice. The conformer receives the canonical speaker embedding in both steps. It is therefore trained to map a synthetic-voice representation toward the canonical-voice representation, which is the core voice conversion objective. This step contributes no gradient to the vocoder; it only updates the frontend (afeat_1d_conv, fusion, Q-Former, projections, conformer).

Total Phase 1 loss: `L_total = L_mel + λ · L_conversion`

Convergence is monitored using:
- **SSIM** (Structural Similarity Index) computed on mel spectrogram images — target value is 1.0
- **MCD** (Mel Cepstral Distortion, in dB) — decreases as the model improves

### Phase 2 — Adversarial Voice Conversion

Once Phase 1 has converged, the discriminator is activated and the training objective shifts entirely to voice conversion:

- **All modules from AV-HuBERT through the conformer are frozen.** Only the vocoder and discriminators are trainable.
- Input: canonical video + **synthetic audio**, canonical speaker embedding
- Ground truth: **canonical audio**
- The model learns to produce perceptually realistic waveforms that match the canonical speaker's voice
- Generator losses: feature-matching loss + adversarial losses from MPD and MS-STFTD
- Discriminator losses: standard GAN hinge losses

The two-phase strategy separates representation learning (Phase 1) from waveform quality optimisation (Phase 2), which stabilises training.

---

## Summary of Tensor Flow (3-second example)

| Stage | Module | Output shape | Notes |
|---|---|---|---|
| Video input | — | `(B, 75, —)` | 75 frames @ 25 fps |
| Audio input | — | `(B, 48000)` | 3 s @ 16 kHz |
| **Video encoder** | AV-HuBERT (frozen) | `(B, 75, 1024)` | Visual-only, 25 Hz |
| **Audio encoder** | Whisper medium (frozen) | `(B, 1500, 1024)` | Padded to 30 s, 50 Hz |
| Audio downsample | afeat_1d_conv | `(B, 750, 1024)` | 50 Hz → 25 Hz |
| Audio trim | slice `[:, :T_v, :]` | `(B, 75, 1024)` | Aligned to video |
| SR predictor input | subsample raw Whisper | `(B, 38, 1024)` | From original 50 Hz features |
| Speech rate | SR predictor (frozen) | scalar ∈ [1, 2] | Determines N_queries |
| Fusion | concat / add / cross-att | `(B, 75, 1024+)` | Modality dropout possible |
| Compression | Q-Former | `(B, N_q, 1024)` | N_q = 9 (queries_per_sec=3, 3s clip), speech-rate adaptive |
| Projector + proj1 + ln1 | MLP + Linear + LN | `(B, N_q, 768)` | |
| Upsampling | Linear interpolation | `(B, 300, 768)` | N_q → mel frame count |
| proj2 + ln2 | Linear + LN | `(B, 300, 512)` | |
| **Latent features** | Conformer + ln3 | `(B, 300, 512)` | Cross-attn to spk_emb per layer |
| **Waveform** | HiFi-GAN vocoder | `(B, 1, 48000)` | 300 × 160 = 48,000 samples |
