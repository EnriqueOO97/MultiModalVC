import os
import random

def split_dataset(tsv_path, wrd_path, output_folder, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1):
    """
    Splits a paired TSV and WRD dataset into train, eval, and test sets.
    """
    
    # 1. Validation
    if abs((train_ratio + eval_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Ratios must sum to 1.0")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    print("Reading files...")

    # 2. Read TSV file
    with open(tsv_path, 'r', encoding='utf-8') as f:
        tsv_lines = f.readlines()
    
    # Extract the root path (first line) and the actual data
    root_path = tsv_lines[0] 
    tsv_data = tsv_lines[1:] # The rest are the actual entries

    # 3. Read WRD file
    with open(wrd_path, 'r', encoding='utf-8') as f:
        wrd_data = f.readlines()

    # 4. Consistency Check
    if len(tsv_data) != len(wrd_data):
        raise ValueError(f"Mismatch in line counts! TSV (excluding root) has {len(tsv_data)}, WRD has {len(wrd_data)}")
    
    total_samples = len(tsv_data)
    print(f"Total samples found: {total_samples}")

    # 5. Zip, Shuffle, and Split
    # We zip them to keep the pair (tsv_line, wrd_line) together during shuffling
    paired_data = list(zip(tsv_data, wrd_data))
    
    # Set a seed for reproducibility (optional, remove if you want random every time)
    random.seed(42) 
    random.shuffle(paired_data)

    # Calculate split indices
    train_end = int(total_samples * train_ratio)
    eval_end = train_end + int(total_samples * eval_ratio)

    train_set = paired_data[:train_end]
    eval_set = paired_data[train_end:eval_end]
    test_set = paired_data[eval_end:]

    print(f"Split sizes -> Train: {len(train_set)}, Eval: {len(eval_set)}, Test: {len(test_set)}")

    # 6. Helper function to write files
    def write_files(dataset_name, data_pairs):
        out_tsv = os.path.join(output_folder, f"{dataset_name}.tsv")
        out_wrd = os.path.join(output_folder, f"{dataset_name}.wrd")

        # Unzip the pairs back into separate lists
        tsv_out_lines, wrd_out_lines = zip(*data_pairs)

        # Write TSV (Remember to add the root_path at the top)
        with open(out_tsv, 'w', encoding='utf-8') as f:
            f.write(root_path) # Write the slash line first
            f.writelines(tsv_out_lines)
        
        # Write WRD
        with open(out_wrd, 'w', encoding='utf-8') as f:
            f.writelines(wrd_out_lines)
        
        print(f"Saved {dataset_name} files.")

    # 7. Save the splits
    write_files("train", train_set)
    write_files("valid", eval_set) # Usually named 'valid' or 'dev' in these frameworks
    write_files("test", test_set)

    print("Done processing.")

# ==========================================
# CONFIGURATION
# ==========================================

# Input file paths
input_tsv_file = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/vox2_german_train_finetuned.tsv"
input_wrd_file = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/vox2_german_train_finetuned.wrd"

# Output directory
output_dir = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest"

# Run the function
split_dataset(input_tsv_file, input_wrd_file, output_dir)