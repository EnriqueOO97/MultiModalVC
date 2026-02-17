import os
import glob
import numpy as np
import librosa
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- DiVISe / HiFi-GAN Parameters ---
SR = 16000
N_FFT = 1024
HOP_LENGTH = 160 
WIN_LENGTH = 1024
N_MELS = 128          
FMIN = 0
FMAX = 8000

# --- Manifest Files ---
MANIFEST_FILES = [
    "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/train.tsv",
    "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/valid.tsv",
    "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/test.tsv"
]

def process_file_wrapper(line):
    """Wrapper to handle errors and parsing within the worker process."""
    try:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            return # Skip invalid lines
        
        # 3rd column is the wav path
        wav_path = parts[2]
        
        # Check if output exists to avoid re-processing
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        output_dir = os.path.dirname(wav_path)
        save_path = os.path.join(output_dir, f"{base_name}_mel_100hz_128bands.pt") # Using _mel.pt to distinguish
        
        if os.path.exists(save_path):
            return 

        process_single_wav(wav_path, save_path)
        
    except Exception as e:
        print(f"Error processing line: {line[:50]}... -> {e}")

def process_single_wav(wav_path, save_path):
    # 1. Load Audio
    # 'sr=SR' ensures resampling if needed
    y, _ = librosa.load(wav_path, sr=SR)
    
    # 2. Pad Audio (Matching HiFi-GAN logic)
    # HiFi-GAN pads the audio so the frames are centered
    pad_size = int((N_FFT - HOP_LENGTH) / 2)
    y = np.pad(y, (pad_size, pad_size), mode='reflect')

    # 3. Generate Mel Spectrogram (Raw Energy)
    # Note: center=False because we manually padded above (standard HiFi-GAN behavior)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window='hann',
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        center=False 
    )

    # 4. LOG COMPRESSION (Natural Log)
    # This matches the 'dynamic_range_compression_torch' from your found code.
    # We clamp to 1e-5 to avoid log(0)
    log_mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))

    # 5. Save (Shape: [80, T]) -> Transpose to [T, 80] if needed by your model, 
    # but standards usually keep [C, T]. DiVISe likely expects [T, C] or [C, T]. 
    # Let's save as [T, 80] which is standard for Transformer inputs (Seq, Dim).
    log_mel_spec = log_mel_spec.T 
    
    tensor = torch.from_numpy(log_mel_spec).float()
    torch.save(tensor, save_path)

def main():
    # --- Prevent Library Threading Oversubscription ---
    # Since we are using multiprocessing, we want each worker to be single-threaded
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    all_lines = []
    print("Reading manifest files...")
    for manifest_path in MANIFEST_FILES:
        print(f"  - {manifest_path}")
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
            # Filter valid lines (those with wav paths)
            valid_lines = [l for l in lines if ".wav" in l]
            all_lines.extend(valid_lines)
            
    print(f"Found {len(all_lines)} files to process.")
    
    # --- Determine Correct CPU Count ---
    # 1. Try SLURM allocation first
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        # SLURM_CPUS_PER_TASK is often just an integer, but checking is safe
        num_processes = int(slurm_cpus)
        print(f"Detected SLURM allocation: {num_processes} CPUs.")
    else:
        # 2. Fallback to physical affinity if possible (sched_getaffinity)
        try:
            num_processes = len(os.sched_getaffinity(0))
            print(f"Detected CPU affinity: {num_processes} CPUs.")
        except AttributeError:
            # 3. Last resort: os.cpu_count() (can be dangerous on shared nodes)
            num_processes = max(1, int(cpu_count() * 0.9))
            print(f"Using 90% of total nodes CPUs: {num_processes} (Warning: might be too high if not whole-node job).")

    print(f"Starting batched processing with {num_processes} workers...")
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress bar
        list(tqdm(pool.imap_unordered(process_file_wrapper, all_lines), total=len(all_lines)))
    
    print("Done!")

if __name__ == "__main__":
    main()