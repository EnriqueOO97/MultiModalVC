import os
import csv
from tqdm import tqdm  # Optional: for a progress bar

# Define the root directory where the 'dev/aac' folder starts
# Based on your previous file path, it seems to be here:
root_dir = "/ceph/shared/ALL/datasets/voxceleb2-V2/dev/aac"
output_csv = "/ceph/shared/ALL/datasets/voxceleb2-V2/voxceleb_audioPaths.csv"

def generate_file_list(root_path, output_file):
    print(f"Scanning directory: {root_path}")
    
    data_rows = []
    
    # Walk through the directory tree
    # We expect structure: root_path -> id_folder -> sequence_folder -> .m4a files
    
    # Get list of IDs to iterate (sorted for consistency)
    if not os.path.exists(root_path):
        print(f"Error: Directory {root_path} does not exist.")
        return

    id_folders = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    
    print(f"Found {len(id_folders)} ID folders. Processing...")

    for id_folder in tqdm(id_folders):
        id_path = os.path.join(root_path, id_folder)
        
        # Get sequence folders inside the ID folder
        seq_folders = sorted([d for d in os.listdir(id_path) if os.path.isdir(os.path.join(id_path, d))])
        
        for seq_folder in seq_folders:
            seq_path = os.path.join(id_path, seq_folder)
            
            # Get all .m4a files in the sequence folder
            files = sorted([f for f in os.listdir(seq_path) if f.endswith('.m4a')])
            
            for file_name in files:
                full_path = os.path.join(seq_path, file_name)
                
                # Append to our data list: ID, Sequence, Full Path
                data_rows.append([id_folder, seq_folder, full_path])

    # Write to CSV
    print(f"Writing {len(data_rows)} entries to {output_file}...")
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['id_folder', 'sequence_folder', 'file_path'])
        # Write data
        writer.writerows(data_rows)
    
    print("Done.")

if __name__ == "__main__":
    generate_file_list(root_dir, output_csv)