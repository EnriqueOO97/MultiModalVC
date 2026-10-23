import os
import glob

def process_manifest(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}...")
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip('\n')
            if not line:
                continue
            parts = line.split('\t')
            
            if len(parts) < 3:
                fout.write(line + '\n')
                continue
                
            audio_path = parts[2]
            audio_dir = os.path.dirname(audio_path)
            audio_basename = os.path.basename(audio_path)
            audio_name_no_ext = os.path.splitext(audio_basename)[0] # e.g., 00090_00
            
            xvector_path = os.path.join(audio_dir, f"{audio_name_no_ext}_xvector.pt")
            
            # Find the 6 synthetic waveforms
            synth_paths = []
            for i in range(1, 7):
                pattern = os.path.join(audio_dir, f"{audio_name_no_ext}_*_s{i}.wav")
                matches = glob.glob(pattern)
                if matches:
                    # just take the first match if multiple (should be 1 anyway)
                    synth_paths.append(matches[0])
                else:
                    # if not found, put a placeholder or missing warning
                    synth_paths.append(f"MISSING_{i}")
                    print(f"Warning: Could not find synthetic file s{i} for {audio_name_no_ext} in {audio_dir}")
            
            # Construct new parts:
            # col 1 (idx 0): dataset
            # col 2 (idx 1): video
            # col 3 (idx 2): audio
            # col 4 (idx 3): xvector
            # col 5-10 (idx 4-9): synths 1 to 6
            # cols 11+ (idx 10+): the rest (frames, samples, duration)
            new_parts = parts[:3] + [xvector_path] + synth_paths + parts[3:]
            
            fout.write('\t'.join(new_parts) + '\n')
            
    print(f"Finished {output_path}")

def main():
    base_dir = '/data/fs201163/eo49197/MultiModalVC/manifest/germanManifest'
    
    manifests = ['train', 'test', 'valid', 'test_inference']
    for split in manifests:
        input_file = os.path.join(base_dir, f'{split}.tsv')
        output_file = os.path.join(base_dir, f'{split}Augmented.tsv')
        if os.path.exists(input_file):
            process_manifest(input_file, output_file)
        else:
            print(f"File not found: {input_file}")

if __name__ == '__main__':
    main()
