#!/usr/bin/env python3
"""How much CONTENT did the QuickVC conversion destroy?

Standalone diagnostic -- touches nothing the eval pipeline uses.

Question: the 856 LRS3 rows added to the balanced manifest have a QuickVC conversion as
the model INPUT (col 2) and the original LRS3 wav as the TARGET (col 3). If those synths
are garbled, training on them teaches the model that English input is unreliable, which
would explain unseen-English WER degrading while German stays flat.

Method: LRS3 ships no ground-truth transcripts, so the ORIGINAL wav's own Whisper
transcription is the reference. For each row:

    ref = ASR(original healthy wav)      <- col 3
    hyp = ASR(quickvc synth wav)         <- col 2
    WER/CER = edit distance between them

A low WER means the synth preserved the words; a high WER means QuickVC destroyed them.
Also reported: the mean ASR confidence (avg logprob) of each side, which catches the case
where BOTH sides transcribe to confident-but-different text.

Normalisation matches scripts/score_asr.py: lowercase, strip punctuation, collapse
whitespace -- so numbers here are comparable to the pipeline's.

Usage:
    python scripts/score_synth_content.py --manifest <tsv> --model openai/whisper-small
"""

import argparse
import csv
import logging
import os
import re
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("synth-content")

DEFAULT_MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
                    "dub-autoalign-healthyYT-Trim-Balanced/1000newLRS3.tsv")


def normalize(s):
    """Same normalisation as scripts/score_asr.py."""
    s = s.lower()
    s = re.sub(r"[^\w äöüßÄÖÜ\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def edits(ref_tokens, hyp_tokens):
    """Levenshtein distance (iterative, two rows)."""
    if not ref_tokens:
        return len(hyp_tokens)
    prev = list(range(len(hyp_tokens) + 1))
    for i, r in enumerate(ref_tokens, 1):
        cur = [i]
        for j, h in enumerate(hyp_tokens, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="9-col manifest; col2 = synth (input), col3 = original (target).")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--language", default="en")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="First N rows (0 = all).")
    ap.add_argument("--out", default="", help="TSV of per-utterance results (optional).")
    args = ap.parse_args()

    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    rows = []
    with open(args.manifest) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            if os.path.isfile(c[1]) and os.path.isfile(c[2]):
                rows.append((c[1], c[2]))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"no usable rows in {args.manifest}")
    logger.info(f"manifest {args.manifest}")
    logger.info(f"{len(rows)} rows with both synth and original present")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[whisper] loading {args.model} -> {device}")
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    forced = proc.get_decoder_prompt_ids(language=args.language, task="transcribe")

    def transcribe(paths):
        """Return (texts, mean avg-logprob) for a list of wav paths."""
        texts, confs = [], []
        for i in range(0, len(paths), args.batch_size):
            chunk = paths[i:i + args.batch_size]
            wavs = []
            for p in chunk:
                w, sr = sf.read(p, dtype="float32")
                if w.ndim > 1:
                    w = w.mean(axis=1)
                if sr != 16000:
                    import librosa
                    w = librosa.resample(w, orig_sr=sr, target_sr=16000)
                wavs.append(w)
            feats = proc(wavs, sampling_rate=16000,
                         return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                out = model.generate(feats, forced_decoder_ids=forced,
                                     return_dict_in_generate=True, output_scores=True)
            seqs = out.sequences
            texts.extend(proc.batch_decode(seqs, skip_special_tokens=True))
            # mean max-logprob per generated step = a crude confidence
            if out.scores:
                lp = torch.stack([s.log_softmax(-1).max(-1).values for s in out.scores], 1)
                confs.extend(lp.mean(dim=1).float().cpu().tolist())
            else:
                confs.extend([float("nan")] * len(chunk))
            if (i // args.batch_size) % 10 == 0:
                logger.info(f"    {min(i + args.batch_size, len(paths))}/{len(paths)}")
        return texts, confs

    logger.info("=== transcribing ORIGINAL (healthy) wavs -> reference")
    ref_txt, ref_conf = transcribe([r[1] for r in rows])
    logger.info("=== transcribing SYNTH (QuickVC) wavs -> hypothesis")
    hyp_txt, hyp_conf = transcribe([r[0] for r in rows])

    w_e = w_n = c_e = c_n = 0
    per, empty_ref, empty_hyp = [], 0, 0
    for (synth, orig), rt, ht, rc, hc in zip(rows, ref_txt, hyp_txt, ref_conf, hyp_conf):
        r, h = normalize(rt), normalize(ht)
        if not r:
            empty_ref += 1
            continue
        if not h:
            empty_hyp += 1
        rw, hw = r.split(), h.split()
        we, cw = edits(rw, hw), len(rw)
        ce, cc = edits(list(r), list(h)), len(r)
        w_e += we; w_n += cw; c_e += ce; c_n += cc
        per.append((os.path.basename(synth), we / max(cw, 1), ce / max(cc, 1),
                    rc, hc, rt.strip(), ht.strip()))

    wers = sorted(p[1] for p in per)
    n = len(per)
    print("\n" + "=" * 72)
    print(f"CONTENT PRESERVATION OF THE QUICKVC SYNTHS   ({args.model})")
    print("=" * 72)
    print(f"  utterances scored     : {n}")
    print(f"  WER corpus (pooled)   : {w_e / max(w_n, 1):.4f}")
    print(f"  CER corpus (pooled)   : {c_e / max(c_n, 1):.4f}")
    print(f"  WER mean / median     : {sum(p[1] for p in per)/n:.4f} / "
          f"{wers[n // 2]:.4f}")
    print(f"  WER p90 / max         : {wers[int(n * 0.9)]:.4f} / {wers[-1]:.4f}")
    print(f"  utts with WER = 0     : {sum(1 for w in wers if w == 0)} "
          f"({100 * sum(1 for w in wers if w == 0) / n:.1f}%)")
    print(f"  utts with WER > 0.5   : {sum(1 for w in wers if w > 0.5)} "
          f"({100 * sum(1 for w in wers if w > 0.5) / n:.1f}%)")
    print(f"  utts with WER > 1.0   : {sum(1 for w in wers if w > 1.0)}")
    print(f"  empty transcription   : {empty_hyp} synth, {empty_ref} original (skipped)")
    rcs = [p[3] for p in per if p[3] == p[3]]
    hcs = [p[4] for p in per if p[4] == p[4]]
    if rcs and hcs:
        print(f"  ASR confidence        : original {np.mean(rcs):+.3f}  "
              f"synth {np.mean(hcs):+.3f}  (higher = more confident)")

    print("\n  --- 5 WORST (most content lost) ---")
    for name, w, c, rc, hc, rt, ht in sorted(per, key=lambda x: -x[1])[:5]:
        print(f"   WER {w:6.2f}  {name}")
        print(f"      original: {rt[:100]}")
        print(f"      synth   : {ht[:100]}")
    print("\n  --- 3 BEST (content preserved) ---")
    for name, w, c, rc, hc, rt, ht in sorted(per, key=lambda x: x[1])[:3]:
        print(f"   WER {w:6.2f}  {name}")
        print(f"      original: {rt[:100]}")
        print(f"      synth   : {ht[:100]}")

    print("\n  INTERPRETATION")
    corpus = w_e / max(w_n, 1)
    if corpus < 0.15:
        print("   Synths PRESERVE content well. They are not the cause of the English")
        print("   degradation -- look at the mixture ratio instead.")
    elif corpus < 0.40:
        print("   Synths carry moderate content damage. Plausible contributor to the")
        print("   English degradation, but probably not the whole story.")
    else:
        print("   Synths are BADLY degraded. Training on them teaches the model that")
        print("   English input is unreliable -- a strong candidate for the cause.")
    print("=" * 72)

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["synth", "WER", "CER", "conf_orig", "conf_synth",
                        "text_orig", "text_synth"])
            for row in sorted(per, key=lambda x: -x[1]):
                w.writerow(row)
        print(f"\nper-utterance TSV -> {args.out}")


if __name__ == "__main__":
    main()
