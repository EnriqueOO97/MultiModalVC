import os
import glob
import torchaudio
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
ROOT_DIR = "/ceph/shared/ALL/datasets/voxceleb2-V2/VoxCeleb2-German/dev/processedVideos/vox2_german"
VIDEO_DIR_NAME = "vox2_german_video_seg16s"
TEXT_DIR_NAME = "vox2_german_text_finetuned_seg16s"

OUTPUT_TSV = "vox2_german_train_finetuned.tsv"
OUTPUT_WRD = "vox2_german_train_finetuned.wrd"

# Number of parallel workers (match your requested CPUs)
NUM_WORKERS = 32 

def process_single_file(mp4_path):
    """
    Worker function to process a single file.
    Returns (tsv_line, wrd_line) or None if failed.
    """
    try:
        wav_path = mp4_path.replace('.mp4', '.wav')
        txt_path = mp4_path.replace(VIDEO_DIR_NAME, TEXT_DIR_NAME).replace('.mp4', '.txt')
        
        if not os.path.exists(wav_path): return None

        # 1. Get Metadata (Audio)
        # torchaudio.info is fast (header only)
        info = torchaudio.info(wav_path)
        n_aud = info.num_frames
        
        # 2. Get Metadata (Video)
        # OpenCV capture is relatively fast for header
        cap = cv2.VideoCapture(mp4_path)
        n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 3. Get Text
        if not os.path.exists(txt_path): return None
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip().lower()
        
        if not text: return None

        # 4. Calculate Rate
        duration = n_aud / 16000.0
        n_words = len(text.split())
        speech_rate = round(n_words / duration, 2) if duration > 0 else 0.0
        
        # 5. Format Lines
        dataset_name = "vox2_german"
        tsv_line = f"{dataset_name}\t{mp4_path}\t{wav_path}\t{n_vid}\t{n_aud}\t{speech_rate}"
        
        return (tsv_line, text)

    except Exception:
        return None

def generate_manifests():
    video_root = os.path.join(ROOT_DIR, VIDEO_DIR_NAME)
    
    print(f"Scanning {video_root}...")
    # Glob is fast enough on single thread usually
    mp4_files = sorted(glob.glob(os.path.join(video_root, "**", "*.mp4"), recursive=True))
    
    print(f"Processing {len(mp4_files)} files with {NUM_WORKERS} workers...")
    
    # Parallel Processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Use tqdm to show progress bar
        results = list(tqdm(executor.map(process_single_file, mp4_files), total=len(mp4_files)))
    
    # Filter out None results (failures)
    valid_results = [r for r in results if r is not None]
    
    print(f"Writing {len(valid_results)} entries...")
    
    # Write TSV
    with open(OUTPUT_TSV, 'w', encoding='utf-8') as f:
        f.write("/\n") 
        for tsv_line, _ in valid_results:
            f.write(tsv_line + "\n")
            
    # Write WRD
    with open(OUTPUT_WRD, 'w', encoding='utf-8') as f:
        for _, wrd_line in valid_results:
            f.write(wrd_line + "\n")
            
    print("Done. Manifests generated.")

if __name__ == "__main__":
    generate_manifests()