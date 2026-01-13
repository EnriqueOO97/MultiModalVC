import torch
import torchaudio
import csv
import os
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

# --- Configuration ---
INPUT_CSV = "/ceph/shared/ALL/datasets/voxceleb2-V2/voxceleb_audioPaths-part2.csv"
OUTPUT_DIR = "/ceph/shared/ALL/datasets/voxceleb2-V2/" # Directory to save results
MODEL_ID = "openai/whisper-large-v3"
BATCH_SIZE = 48  # Per GPU
NUM_WORKERS = 4  # Per GPU

# --- 1. Dataset (Same as before) ---
class AudioDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio_path = row['file_path']
        try:
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.squeeze(0) 
            if waveform.shape[0] > 480000:
                waveform = waveform[:480000]
            return {"audio": waveform.numpy(), "path": audio_path, "valid": True}
        except Exception:
            return {"audio": None, "path": audio_path, "valid": False}

def collate_fn(batch):
    # We need the processor, but it's inside the model. 
    # We will handle tokenization inside the LightningModule for simplicity in DDP
    # or we can initialize a processor globally. Global is easier here.
    return batch 

# --- 2. Lightning Module ---
class WhisperPredictor(pl.LightningModule):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        # We load model/processor in setup() or init. 
        # For DDP, it's safe to load in __init__ as Lightning handles moving to device.
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        # We don't need model.to(device), Lightning handles it.

    def forward(self, inputs):
        # Not used directly in predict_step usually, but good practice
        pass

    def predict_step(self, batch, batch_idx):
        # 1. Unpack batch (custom collate logic moved here or done before)
        # Since we didn't tokenize in collate, we do it here.
        valid_items = [item for item in batch if item['valid']]
        invalid_paths = [item['path'] for item in batch if not item['valid']]
        
        if not valid_items:
            return {"valid": [], "invalid": invalid_paths}

        audio_arrays = [item['audio'] for item in valid_items]
        valid_paths = [item['path'] for item in valid_items]

        # Tokenize
        inputs = self.processor(
            audio_arrays, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True
        )
        
        # Move to device (Lightning handles self.device)
        input_features = inputs.input_features.to(self.device, dtype=self.model.dtype)

        # Generate
        generated_ids = self.model.generate(
            input_features, 
            max_new_tokens=1,
            return_dict_in_generate=False
        )

        decoded_outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        
        results = []
        for path, output_str in zip(valid_paths, decoded_outputs):
            lang = self.extract_language_token(output_str)
            results.append((path, lang))
            
        return {"valid": results, "invalid": invalid_paths}

    def extract_language_token(self, decoded_str):
        if "<|" in decoded_str:
            parts = decoded_str.split("|>")
            for p in parts:
                if "<" in p:
                    token = p.split("<|")[-1]
                    if len(token) == 2: 
                        return token
        return "unknown"

# --- 3. Custom Writer Callback ---
class CSVWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # Each GPU (rank) writes to its own file to avoid locking issues
        rank = trainer.global_rank
        output_file = os.path.join(self.output_dir, f"results_rank_{rank}.csv")
        
        file_exists = os.path.isfile(output_file)
        
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['file_path', 'detected_language'])
            
            # Write valid results
            for path, lang in prediction["valid"]:
                writer.writerow([path, lang])
            
            # Write errors
            for path in prediction["invalid"]:
                writer.writerow([path, "error_loading"])

# --- 4. Main Execution ---
if __name__ == "__main__":

    keys_to_unset = ["SLURM_NTASKS", "SLURM_JOB_NAME", "SLURM_JOB_ID", "SLURM_NPROCS"]
    for key in keys_to_unset:
        if key in os.environ:
            del os.environ[key]

    start_time = time.time()
    
    # 1. Read CSV
    print("Reading input CSV...")
    with open(INPUT_CSV, 'r', encoding='utf-8') as f_in:
        rows = list(csv.DictReader(f_in))
        
    # 2. Setup Data
    dataset = AudioDataset(rows)
    # Note: No sampler needed here, Lightning adds DistributedSampler automatically!
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn # Simple identity collate
    )

    # 3. Setup Lightning Components
    model = WhisperPredictor(MODEL_ID)
    writer = CSVWriter(OUTPUT_DIR, write_interval="batch")
    
    # 4. Trainer
    # devices=2 will use 2 GPUs. strategy="ddp" enables parallel processing.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2, # Uses all available GPUs (e.g., 2)
        strategy="ddp", # Distributed Data Parallel
        callbacks=[writer],
        enable_progress_bar=True,
        logger=False, # Disable logging for cleaner output
        precision="16-mixed"
    )

    print(f"Starting inference on {trainer.num_devices} GPUs...")
    trainer.predict(model, dataloader)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")