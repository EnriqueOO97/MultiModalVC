import pandas as pd
import os
import glob

# --- Configuration ---
# The directory where your results_rank_0.csv and results_rank_1.csv are located
OUTPUT_DIR = "/ceph/shared/ALL/datasets/voxceleb2-V2/"
FINAL_FILENAME = "voxceleb_language_results_merged.csv"

def merge_csvs():
    print("Looking for rank files...")
    # Find all files matching the pattern
    file_pattern = os.path.join(OUTPUT_DIR, "results_rank_*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print("No result files found!")
        return

    print(f"Found {len(files)} files: {files}")
    
    # 1. Load all CSVs into a list of DataFrames
    dfs = []
    for f in files:
        print(f"Reading {f}...")
        dfs.append(pd.read_csv(f))

    # 2. Concatenate them into one big table
    print("Concatenating...")
    full_df = pd.concat(dfs, ignore_index=True)

    # 3. Sort by file_path
    # Since your original dataset seems to be ordered by path (00126, 00127...), 
    # sorting by this column will perfectly reconstruct the original order.
    print("Sorting by file_path...")
    full_df = full_df.sort_values(by="file_path")

    # 4. Save final result
    output_path = os.path.join(OUTPUT_DIR, FINAL_FILENAME)
    print(f"Saving {len(full_df)} rows to {output_path}...")
    full_df.to_csv(output_path, index=False)
    
    print("Done!")

if __name__ == "__main__":
    merge_csvs()