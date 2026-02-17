
import sys
import os
import torch

# Add fairseq to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'fairseq'))

def main():
    ckpt_path = 'pretrained_models/mms_llama/1759h/ckpt-1759h.pt'
    try:
        print(f"Loading {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        keys = list(ckpt['model'].keys())
        
        has_avhubert = any('encoder.w2v_model' in k for k in keys)
        has_whisper = any('whisper' in k for k in keys)
        
        has_avhubert_real = any('avhubert.w2v_model' in k for k in keys)
        has_whisper_real = any('whisper' in k for k in keys)
        
        print(f"Has AV-HuBERT keys (avhubert.w2v_model.*): {has_avhubert_real}")
        print(f"Has Whisper keys (whisper.*): {has_whisper_real}")
        
        if not has_whisper_real:
            print("Whisper keys confirm MISSING.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
