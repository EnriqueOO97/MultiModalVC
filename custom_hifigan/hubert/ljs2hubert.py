import argparse
from pathlib import Path
import soundfile as sf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_dir", metavar="in-dir", help="path to resampled 16k dataset directory.", type=Path
    )
    parser.add_argument(
        "out_dir", metavar="out-dir", help="path to the output directory.", type=Path
    )
    args = parser.parse_args()
    for split in ["validation", "training"]:
        input_txt = args.in_dir / f"{split}.txt"
        output_tsv = args.out_dir / f"{split}.tsv"
        with open(input_txt, 'r') as f:
            all_lines = f.readlines()
        all_lines = [args.in_dir / 'wavs'/ Path(str(x.split('|')[0])+'.wav') for x in all_lines]
        with open(output_tsv, 'w') as f:
            f.write("/\n")
            for wav_path in all_lines:
                wav, sr = sf.read(wav_path)
                assert sr == 16000, sr
                f.write(str(wav_path)+"\t"+str(len(wav))+"\n")