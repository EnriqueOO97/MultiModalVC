import os
import argparse
import pandas as pd
import numpy as np

def normalize_speech_rates(tsv_path, output_path):
    print(f"Processing: {tsv_path}")
    
    # Check if file exists
    if not os.path.exists(tsv_path):
        print(f"Error: File not found at {tsv_path}")
        return

    # 1. Read the TSV file
    # We assume standard CSV/TSV parsing. 
    # The first line is usually just "/" as a root indicator in this dataset format, 
    # but the subsequent lines have tab-separated values.
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if not lines:
        print("Error: File is empty.")
        return

    header = lines[0].strip() # Should be "/"
    data_lines = lines[1:]
    
    # Parse data lines
    parsed_data = []
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 6: # We expect at least 6 columns, with the 6th being speech rate
            # columns: id, vid_path, aud_path, n_frames, n_samples, speech_rate
            parsed_data.append(parts)
        else:
            print(f"Warning: Skipping malformed line (cols={len(parts)}): {line.strip()}")

    if not parsed_data:
        print("Error: No valid data found in file.")
        return

    # 2. Extract Speech Rates
    # Column index 5 is the 6th column (0-indexed)
    speech_rates = []
    for row in parsed_data:
        try:
            rate = float(row[5])
            speech_rates.append(rate)
        except ValueError:
            print(f"Warning: Could not parse speech rate '{row[5]}'")
            speech_rates.append(0.0)

    # 3. Compute Mean
    mean_rate = np.mean(speech_rates)
    print(f"Total samples: {len(speech_rates)}")
    print(f"Mean Speech Rate: {mean_rate:.4f} words/sec")

    if mean_rate == 0:
        print("Error: Mean speech rate is 0. Cannot normalize.")
        return

    # 4. Normalize and Create New Data
    new_lines = [header + "\n"]
    
    for row, raw_rate in zip(parsed_data, speech_rates):
        normalized_rate = raw_rate / mean_rate
        # Reconstruct the line
        # We replace the 6th column with the formatted normalized rate
        new_row = row[:5] + [f"{normalized_rate:.2f}"] + row[6:] # In case there are more columns
        new_line = "\t".join(new_row) + "\n"
        new_lines.append(new_line)

    # 5. Write Output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Successfully wrote normalized TSV to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize speech rates in a TSV manifest.")
    parser.add_argument("--input", type=str, required=True, help="Input TSV file")
    parser.add_argument("--output", type=str, required=True, help="Output TSV file")
    
    args = parser.parse_args()
    
    normalize_speech_rates(args.input, args.output)
