# MelVC: model checkpoints, test manifests and evaluation code

MelVC converts pathological speech into healthy-sounding speech using both the audio and the lip video. It predicts an 80-band mel spectrogram (a time-frequency picture of the speech). A separate BigVGAN vocoder (neural network that turns mel spectrograms into audio) then turns it into a waveform.

## 1. Model checkpoints

The best models are weight-space merges (WiSE-FT, weight-space ensembling for fine-tuning) of a finetuned checkpoint and a pretrained one:
`merged = alpha * finetuned + (1 - alpha) * pretrained`.

| Model | Path |
|---|---|
| **Best: balanced-700 x ck73, alpha 0.9** | `/data/fs201163/eo49197/wiseft_checkpoints/balanced700_ck73/wiseft700ck73_a0.9.pt` |
| balanced-700 x ck73, alpha 0.8 ("checkpoint C") | `/data/fs201163/eo49197/wiseft_checkpoints/balanced700_ck73/wiseft700ck73_a0.8.pt` |
| Other alphas (0.85, 0.95) | same folder, `wiseft700ck73_a0.85.pt`, `wiseft700ck73_a0.95.pt` |
| Alternative: cont50 x ck73 (alpha 0.8 to 0.95) | `/data/fs201163/eo49197/wiseft_checkpoints/cont50_ck73/wiseftCont50ck73_a{0.8,0.85,0.9,0.95}.pt` |
| Finetuned parent (balanced-700, unmerged) | `/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_gamma5_balanced/checkpoints/checkpoint700.pt` |
| Pretrained parent (ck73) | `/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_de_warp_softdtw_adv2/checkpoints/checkpoint73.pt` |
| Vocoder (BigVGAN generator) | `/data/fs201163/eo49197/MultiModalVC/exp/bigvganGTA/p2h_from_de_stage2_OGM_ck500/g_best_mrstft` |

`/data/fs201163/eo49197/wiseft_bal700_ck73/checkpoints/checkpoint_a*.pt` are symlinks to the same merged files.

## 2. Test manifests

All files are in `/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT-trim/` unless a full path is given.

| Test set | Manifest | Transcripts (for word error rate) | Utterances | ASR language |
|---|---|---|---|---|
| In-domain (German) | `testPATH-HE.tsv` | `/data/fs201163/eo49197/VoiceConversion-fwf/transcripts.csv` | 1608 | `de` |
| Unseen speakers, German | `unseenSpeakers.tsv` | `transcripts_unseen.csv` | 240 | `de` |
| Unseen speakers, English | `unseenSpeakersEN.tsv` | `transcripts_unseen_en.csv` | 187 | `en` |

The manifest is tab-separated with 9 columns. The three that matter for evaluation are:

- `video_path`: the lip video (`*_muted.mp4`)
- `pathological_path`: the **input** wav
- `file_path`: the healthy **target** wav. The target speaker embedding is taken from this file.

The remaining six columns (`synth1..6`) are only used during training.

Output files are named by a uid built from the input's folder (`0017-CO-shortstories_0008` = source speaker `0017`, target `CO`, utterance `shortstories_0008`). Every scorer uses this uid to match an output back to its manifest row.

## 3. Evaluation pipeline

| Script (in `/data/fs201163/eo49197/MultiModalVC/scripts/`) | Role |
|---|---|
| `run_eval_pipeline.py` | In-domain driver. Set the checkpoint list (`CHECKPOINTS`), vocoder, manifest and output folder at the top of the file |
| `run_eval_pipeline_unseen.py` | Same driver for the unseen sets. `MANIFEST`, `TRANSCRIPTS` and `LANGUAGE` can be set as environment variables |
| `run_eval_pipeline.sh` | SLURM (cluster job scheduler) wrapper for the driver |
| `evaluate_melvc.py` | `load_model_and_cfg()` shows **how to load a checkpoint** (fairseq `load_model_ensemble_and_task`) |
| `dump_mels_eval.py` | Stage 1: runs inference and writes one `.npy` mel per utterance. Contains the inference modes |
| `dump_wavs_eval.py` | Stage 2: vocodes the mels into wavs |
| `score_mos.py` | Stage 3: DNSMOS Pro and UTMOS (neural quality scores, no reference needed) |
| `score_secs.py` | Stage 4: speaker similarity (ECAPA-TDNN speaker-embedding cosine to the target wav) |
| `score_asr.py` | Stage 5: word and character error rate with `openai/whisper-large-v3` |

Model and data code: `src/modelSpeechNoLLM_E2E_SynthVC.py` (model), `src/dataset_pathological_finetune.py` (manifest loading, noise injection).

### Inference modes (environment variables read by the driver)

| Variable | Values | Meaning |
|---|---|---|
| `MODE` | `dp` (default) | duration prediction: the model decides the output length |
| | `tf` | teacher forcing: output length forced to the **input** length |
| `MODALITY` | `av` (default), `audio_only`, `video_only` | which input streams are used |
| `NOISE_WAV`, `NOISE_PROB`, `SNR`, `NOISE_SEED` | e.g. babble noise file, `1.0`, `0`, `1234` | optional noise added to the input audio only. SNR (signal-to-noise ratio) is in dB |
| `OUT_ROOT` | folder | where mels, wavs and scores are written |

The reported results use `MODE=dp MODALITY=av`.

### Example

```
cd /data/fs201163/eo49197/MultiModalVC
OUT_ROOT=/data/fs201163/eo49197/DumpedMels_example MODE=dp MODALITY=av sbatch scripts/run_eval_pipeline.sh
```

The scores end up in each wav folder: `asr_whisper-large-v3.txt` (its `CORPUS` row holds corpus WER and CER), `secs_scores.txt` and `mos_scores.txt`.

### Reference numbers (best model, in-domain, `dp` + `av`)

Word error rate 0.180, character error rate 0.088, speaker similarity 0.64, DNSMOS Pro 2.80, UTMOS 3.33. For comparison, the unprocessed pathological input has a word error rate of 0.314.
