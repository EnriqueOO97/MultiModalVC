"""
Eval pipeline wrapper: Stage 1 (dump mels) + Stage 2 (vocode to wavs), inline spec.

Stage 1: for each checkpoint, dump predicted mels in the requested inference mode
         ('tf' teacher forcing / 'dp' duration prediction) and modality (set by the
         MODALITY constant: 'av' | 'audio_only' | 'video_only'), via
         scripts/dump_mels_eval.py. Output:
             OUT_ROOT/<run>/<run>-<ckpt_id>-npy/<uid>.npy
Stage 2: vocode EVERY npy folder with EVERY vocoder (so N_ckpt x N_vocoder wav folders),
         via scripts/dump_wavs_eval.py. Each wav folder is named
             <npy_folder_name>__<vocoder_tag>
         where vocoder_tag = the generator's PARENT folder name (all generators are
         called g_best/g_best_mel, so the parent folder is what distinguishes them).

Stage 3: neural MOS (DNSMOS + UTMOS) over every wav folder -> mos_scores.txt inside each,
         via scripts/score_mos.py.
Stage 4: speaker similarity (SECS, ECAPA cosine) over every wav folder -> secs_scores.txt
         inside each, via scripts/score_secs.py (paired to MANIFEST by the uid naming).
Stage 5: ASR intelligibility (WER/CER, Whisper-small) via scripts/score_asr.py:
         predictions -> asr_scores.txt inside each wav folder; plus the roof (healthy GT)
         and floor (pathological input) baselines written once to OUT_ROOT as
         asr_healthy.txt / asr_source.txt.

Everything is inline below — edit CHECKPOINTS / VOCODERS / MANIFEST / OUT_ROOT to change
the run. Generated JSON configs are saved under OUT_ROOT/_pipeline_configs for traceability.

    python scripts/run_eval_pipeline.py [--stage all|1,2,3,4,5] [--overwrite] [--batch-size 32]
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

# ----------------------------------------------------------------------------- spec
MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
            "dub-autoalign-healthyYT-trim/testPATH-HE.tsv")
TRANSCRIPTS = "/data/fs201163/eo49197/VoiceConversion-fwf/transcripts.csv"
# OUT_ROOT and MODALITY are env-overridable so two modality runs (av / audio_only) can be
# launched from the SAME file with no edit-between-jobs race:
#   OUT_ROOT=..._avtf MODALITY=av         sbatch scripts/run_eval_pipeline.sh
#   OUT_ROOT=..._aotf MODALITY=audio_only sbatch scripts/run_eval_pipeline.sh
OUT_ROOT = os.environ.get("OUT_ROOT", "/data/fs201163/eo49197/DumpedMels_run5_tf_av")
MODALITY = os.environ.get("MODALITY", "av")   # 'av' | 'audio_only' | 'video_only'

# Optional noisy-input eval (Stage 1 only): inject babble noise on the INPUT audio (target
# stays clean). Env-overridable, same pattern as OUT_ROOT/MODALITY. Empty NOISE_WAV = clean.
#   NOISE_WAV=/.../babble_noise.wav NOISE_PROB=1.0 SNR=0 OUT_ROOT=... MODALITY=... sbatch ...
NOISE_WAV = os.environ.get("NOISE_WAV", "")   # "" -> clean input (default)
NOISE_PROB = float(os.environ.get("NOISE_PROB", "0.0"))
SNR = os.environ.get("SNR", "")               # e.g. "0" or "-10,-5,0"; blank if clean
NOISE_SEED = os.environ.get("NOISE_SEED", "1234")   # reproducible eval noise
MODE = os.environ.get("MODE", "dp")   # 'dp' = duration prediction | 'tf' = teacher forcing

# ASR (Stage 5) model. large-v3 gave the best (lowest) ceiling, so it's the standard here.
# The per-folder ASR report is model-tagged (asr_<model>.txt) so switching ASR models
# never clobbers or reuses another model's results -- everything recomputes cleanly per model.
ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3")

# (checkpoint, mode): mode 'tf' = teacher forcing, 'dp' = duration prediction.
# Modality x mode ablation on the test split (MODALITY and MODE are both env-overridable).
_EXP = "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth"
_CONT = f"{_EXP}/melvc_p2h_warp_softdtw_adv2_sweep_cont/checkpoints"
_G01 = f"{_EXP}/melvc_gamma01/checkpoints"
_G5 = f"{_EXP}/melvc_gamma5/checkpoints"
_G5B = f"{_EXP}/melvc_gamma5_balanced/checkpoints"
# Continuation of the balanced run (init from balanced ck800) on the manifest that now
# carries 856 QuickVC-converted -> original LRS3 rows (9,978 rows total).
# Alpha sweep on the cont50 x ck73 pair. alpha = share of the FINETUNED (cont50) model.
# Measured drift: cont50 sits 0.091 from ck73 but only 0.027 from balanced-700, so at
# alpha=0.8 the pull toward ck73 (~0.018) swamps the continuation's gain -- which is why
# a0.8 landed on checkpoint C's numbers. Higher alpha keeps more of cont50.
# a0.8 is already scored (DumpedMels_cont50ck73_*); these are the new points.
# Noise robustness sweep for balanced-700 x ck73 at alpha=0.9. SNR is env-overridable
# (NOISE_WAV / NOISE_PROB / SNR), so the whole sweep runs off THIS spec -- one launch per
# (SNR, modality) cell, no edit between jobs. Babble noise is injected on the INPUT audio
# only; the target stays clean.
_BC = "/data/fs201163/eo49197/wiseft_bal700_ck73/checkpoints"
CHECKPOINTS = [
    (f"{_BC}/checkpoint_a0.9.pt", MODE),
]

# 6 BigVGAN generators. Same-folder files share a parent dir, so the wav-folder tag
# (_vocoder_tag) uses '<parent_dir>__<gen_filename>' to keep them all distinct.
_GTA = "/data/fs201163/eo49197/MultiModalVC/exp/bigvganGTA"
VOCODERS = [
    f"{_GTA}/p2h_from_de_stage2_OGM_ck500/g_best_mrstft",   # best vocoder in run3
]

MODE_MAP = {"tf": "tf", "dp": "free"}    # our names -> dump_mels_eval.py mode names
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ASR_TAG = ASR_MODEL.split("/")[-1]       # e.g. whisper-large-v3 -> used in ASR output names
# -----------------------------------------------------------------------------


def _run_name(ckpt):
    return os.path.basename(os.path.dirname(os.path.dirname(ckpt)))  # <run>/checkpoints/x.pt


def _ckpt_id(ckpt):
    stem = os.path.splitext(os.path.basename(ckpt))[0]               # checkpoint250 / checkpoint_best
    return re.sub(r"^checkpoint_?", "", stem) or stem


def _npy_folder(ckpt):
    """MUST match dump_mels_eval.py: OUT_ROOT/<run>/<run>-<ckpt_id>-npy (single-combo name)."""
    run = _run_name(ckpt)
    return os.path.join(OUT_ROOT, run, f"{run}-{_ckpt_id(ckpt)}-npy")


def _vocoder_tag(generator):
    # '<parent_dir>__<gen_filename>' so multiple generator files from the SAME vocoder
    # folder (e.g. g_best_mel, g_best_mrstft, g_best_pesq) stay distinct in the wav-folder
    # name -- the vocoder analogue of _ckpt_id disambiguating same-run generator checkpoints.
    return f"{os.path.basename(os.path.dirname(generator))}__{os.path.basename(generator)}"


def _cfg_dir():
    d = os.path.join(OUT_ROOT, "_pipeline_configs")
    os.makedirs(d, exist_ok=True)
    return d


def build_stage1_config(path):
    entries = []
    for ckpt, mode in CHECKPOINTS:
        if mode not in MODE_MAP:
            raise ValueError(f"unknown mode {mode!r} for {ckpt} (use 'tf' or 'dp')")
        entries.append({
            "checkpoint": ckpt,
            "modes": [MODE_MAP[mode]],
            "modalities": [MODALITY],
        })
    json.dump(entries, open(path, "w"), indent=2)
    return entries


def build_stage2_config(path):
    npy_folders = [_npy_folder(ckpt) for ckpt, _ in CHECKPOINTS]
    jobs = []
    for gen in VOCODERS:
        jobs.append({
            "generator": gen,
            "tag": _vocoder_tag(gen),
            "npy_folders": npy_folders,          # every vocoder runs over every npy folder
        })
    json.dump(jobs, open(path, "w"), indent=2)
    return jobs


def run(cmd):
    print("[wrapper] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def banner(msg):
    print("\n" + "=" * 72 + f"\n[wrapper] {msg}\n" + "=" * 72, flush=True)


def preflight(stages):
    """Fail fast (before any GPU work) if inputs a stage needs are missing, with an
    explicit list of what's wrong so the log points straight at the problem."""
    errs = []
    if not os.path.isfile(MANIFEST):
        errs.append(f"manifest missing: {MANIFEST}")
    if 1 in stages:
        for ckpt, _ in CHECKPOINTS:
            if not os.path.isfile(ckpt):
                errs.append(f"checkpoint missing: {ckpt}")
    if 2 in stages:
        for gen in VOCODERS:
            if not os.path.isfile(gen):
                errs.append(f"generator missing: {gen}")
            cfg = os.path.join(os.path.dirname(gen), "config.json")
            if not os.path.isfile(cfg):
                errs.append(f"vocoder config.json missing: {cfg}")
    if 5 in stages and not os.path.isfile(TRANSCRIPTS):
        errs.append(f"transcripts missing: {TRANSCRIPTS}")
    if 1 in stages and NOISE_WAV and NOISE_PROB > 0.0 and not os.path.isfile(NOISE_WAV):
        errs.append(f"noise wav missing: {NOISE_WAV}")
    # Stages 3/4/5-pred consume the Stage-2 wav folders; if Stage 2 isn't in THIS run
    # they must already exist on disk.
    if (stages & {3, 4, 5}) and 2 not in stages:
        wav_dirs = [f"{_npy_folder(c)}__{_vocoder_tag(g)}"
                    for c, _ in CHECKPOINTS for g in VOCODERS]
        missing = [d for d in wav_dirs if not os.path.isdir(d)]
        if missing:
            errs.append(f"{len(missing)}/{len(wav_dirs)} Stage-2 wav folders missing "
                        f"(run stage 2 first); e.g. {missing[0]}")
    if errs:
        banner("PREFLIGHT FAILED")
        for e in errs:
            print("   - " + e, flush=True)
        sys.exit(1)
    print(f"[wrapper] preflight OK  (stages={sorted(stages)}, "
          f"{len(CHECKPOINTS)} checkpoints, {len(VOCODERS)} vocoders)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="Which stages: 'all' (1-5) or a comma list, e.g. '3,4,5'. "
                         "1=dump mels, 2=vocode, 3=MOS, 4=SECS, 5=ASR (WER/CER).")
    ap.add_argument("--batch-size", type=int, default=32, help="Stage 2 vocoder batch size.")
    ap.add_argument("--score-batch-size", type=int, default=48,
                    help="Batch size for the batched scorers (Stage 4 SECS, Stage 5 ASR).")
    ap.add_argument("--num-workers", type=int, default=8, help="Stage 1 dataloader workers.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ow = ["--overwrite"] if args.overwrite else []
    stages = ({1, 2, 3, 4, 5} if args.stage.strip() == "all"
              else {int(x) for x in args.stage.split(",")})

    noise_desc = (f"noise=ON wav={os.path.basename(NOISE_WAV)} prob={NOISE_PROB} snr={SNR or '?'}"
                  if (NOISE_WAV and NOISE_PROB > 0.0) else "noise=OFF (clean input)")
    banner(f"EVAL PIPELINE START  stages={sorted(stages)}  out_root={OUT_ROOT}  "
           f"modality={MODALITY}  {noise_desc}")
    preflight(stages)

    # Every Stage-2 wav folder: <npy_folder>__<vocoder_tag>. Stages 3/4/5 score all of them.
    wav_dirs = [f"{_npy_folder(ckpt)}__{_vocoder_tag(gen)}"
                for ckpt, _ in CHECKPOINTS for gen in VOCODERS]
    pipe_t0 = time.time()

    def stage(num, name, fn):
        if num not in stages:
            return
        banner(f"STAGE {num} START — {name}")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            banner(f"STAGE {num} FAILED after {time.time() - t0:.1f}s — {name}  "
                   f"({type(e).__name__}: {e})")
            raise
        banner(f"STAGE {num} DONE in {time.time() - t0:.1f}s — {name}")

    def _s1():
        c1 = os.path.join(_cfg_dir(), f"stage1_{ts}.json")
        build_stage1_config(c1)
        noise_args = []
        if NOISE_WAV and NOISE_PROB > 0.0:
            noise_args = ["--noise-wav", NOISE_WAV, "--noise-prob", str(NOISE_PROB),
                          "--noise-seed", NOISE_SEED]
            if SNR:
                noise_args += ["--snr", SNR]
        run([sys.executable, os.path.join(SCRIPTS, "dump_mels_eval.py"),
             "--config", c1, "--manifest", MANIFEST, "--out-root", OUT_ROOT,
             "--num-workers", str(args.num_workers)] + noise_args + ow)

    def _s2():
        c2 = os.path.join(_cfg_dir(), f"stage2_{ts}.json")
        build_stage2_config(c2)
        run([sys.executable, os.path.join(SCRIPTS, "dump_wavs_eval.py"),
             "--config", c2, "--batch-size", str(args.batch_size)] + ow)

    def _s3():
        run([sys.executable, os.path.join(SCRIPTS, "score_mos.py"), *wav_dirs] + ow)
        # Reference-free quality needs no transcripts and no target, so the floor (the
        # pathological INPUT) and the healthy target can be scored too -> mos_<role>.txt
        # in OUT_ROOT/mos_baseline/, same role pattern as the ASR baselines in _s5.
        mos = os.path.join(SCRIPTS, "score_mos.py")
        base_dir = os.path.join(OUT_ROOT, "mos_baseline")
        for role in ("source", "healthy"):                  # floor, ceiling
            run([sys.executable, mos, "--role", role, "--manifest", MANIFEST,
                 "--out-dir", base_dir] + ow)

    def _s4():
        run([sys.executable, os.path.join(SCRIPTS, "score_secs.py"), *wav_dirs,
             "--manifest", MANIFEST, "--batch-size", str(args.score_batch_size)] + ow)

    def _s5():
        asr = os.path.join(SCRIPTS, "score_asr.py")
        common = ["--manifest", MANIFEST, "--transcripts", TRANSCRIPTS,
                  "--model", ASR_MODEL, "--batch-size", str(args.score_batch_size)]
        # Model-tagged output names so switching ASR models never reuses/clobbers another
        # model's numbers: pred -> asr_<model>.txt inside each wav folder; baselines ->
        # asr_<role>.txt inside OUT_ROOT/asr_baseline_<model>/.
        pred_name = f"asr_{ASR_TAG}.txt"
        base_dir = os.path.join(OUT_ROOT, f"asr_baseline_{ASR_TAG}")
        run([sys.executable, asr, *wav_dirs, "--role", "pred",
             "--out-name", pred_name] + common + ow)                        # predictions
        for role in ("healthy", "source"):                                  # roof + floor
            run([sys.executable, asr, "--role", role, "--out-dir", base_dir] + common + ow)

    stage(1, f"dump mels ({len(CHECKPOINTS)} checkpoints -> {OUT_ROOT})", _s1)
    stage(2, f"vocode ({len(VOCODERS)} vocoders x {len(CHECKPOINTS)} npy = "
             f"{len(wav_dirs)} wav folders)", _s2)
    stage(3, f"MOS ({len(wav_dirs)} wav folders)", _s3)
    stage(4, f"SECS ({len(wav_dirs)} wav folders)", _s4)
    stage(5, f"ASR/WER-CER ({len(wav_dirs)} wav folders + healthy/source baselines)", _s5)

    banner(f"EVAL PIPELINE ALL DONE in {time.time() - pipe_t0:.1f}s")


if __name__ == "__main__":
    main()
