# import pandas as pd
# import os

# # --- CONFIGURATION ---
# INPUT_FILE = '/ceph/shared/ALL/datasets/voxceleb2-V2/audio_clips_meta_data.csv'  # Change this to your input file path
# OUTPUT_FOLDER = '/ceph/shared/ALL/datasets/voxceleb2-V2'   # Change this to your desired output folder
# OUTPUT_FILENAME = 'german_files.csv'
# # ---------------------

# def filter_german_files():
#     # 1. Create output path
#     if not os.path.exists(OUTPUT_FOLDER):
#         os.makedirs(OUTPUT_FOLDER)
#         print(f"Created folder: {OUTPUT_FOLDER}")
    
#     output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    
#     print(f"Processing {INPUT_FILE}...")
    
#     # 2. Process in chunks (safe for massive files)
#     chunk_size = 100000  # Process 100k rows at a time
#     first_chunk = True
#     total_rows = 0

#     # We read the file in parts
#     for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size):
        
#         # Filter for German language
#         german_rows = chunk[chunk['language'] == 'de']
        
#         if not german_rows.empty:
#             # Write to file
#             # mode='a' means append, header=first_chunk ensures header is only written once
#             german_rows.to_csv(output_path, mode='a', index=False, header=first_chunk)
            
#             total_rows += len(german_rows)
#             first_chunk = False
#             print(f"Found and saved {len(german_rows)} rows...", end='\r')

#     print(f"\nDone! Total German rows saved: {total_rows}")
#     print(f"File saved at: {output_path}")

# if __name__ == "__main__":
#     filter_german_files()

#########################################################################################################################################################
#                                                     Distinguish between dev and test speakers

# import pandas as pd
# import os

# # --- Configuration ---
# # Path to the file you just created with the German audio clips
# german_clips_path = "/ceph/shared/ALL/datasets/voxceleb2-V2/german_files.csv"

# # Path to the metadata file (the one with Name, Gender, etc.)
# metadata_path = "/ceph/shared/ALL/datasets/voxceleb2-V2/vox2_meta.csv" 

# # Output path
# output_path = "/ceph/shared/ALL/datasets/voxceleb2-V2/vox2_meta_german.csv"

# def extract_german_metadata():
#     # 1. Load the German clips file to get the list of Speaker IDs
#     if not os.path.exists(german_clips_path):
#         print(f"Error: File not found at {german_clips_path}")
#         return

#     df_german_clips = pd.read_csv(german_clips_path)
    
#     # Get unique speaker IDs. Assuming the column is named 'speaker_id' based on previous context.
#     # If it's different, change 'speaker_id' below.
#     unique_german_ids = df_german_clips['speaker_id'].unique()
#     print(f"Found {len(unique_german_ids)} unique German speaker IDs in the clips file.")

#     # 2. Load the Metadata file
#     # FIX: Using sep=r'\s+' to handle tabs or multiple spaces as delimiters
#     try:
#         df_meta = pd.read_csv(metadata_path, sep=r'\s+')
#     except Exception as e:
#         print(f"Error reading metadata file: {e}")
#         return

#     print(f"Loaded metadata file with {len(df_meta)} rows.")
    
#     # Clean up column names (remove trailing whitespace if any)
#     df_meta.columns = df_meta.columns.str.strip()

#     # 3. Filter the metadata
#     # We filter where 'VoxCeleb2 ID' matches our list of German IDs
#     # Ensure both are strings to avoid type mismatch errors
#     unique_german_ids = unique_german_ids.astype(str)
#     df_meta['VoxCeleb2_ID'] = df_meta['VoxCeleb2_ID'].astype(str)

#     german_metadata = df_meta[df_meta['VoxCeleb2_ID'].isin(unique_german_ids)]

#     # 4. Check for missing speakers
#     found_ids = german_metadata['VoxCeleb2_ID'].unique()
#     missing_ids = set(unique_german_ids) - set(found_ids)

#     print(f"Successfully extracted {len(german_metadata)} rows.")
    
#     if len(missing_ids) == 0:
#         print("SUCCESS: All German speakers were found in the metadata file.")
#     else:
#         print(f"WARNING: {len(missing_ids)} speaker IDs were NOT found in the metadata file.")
#         print("Here are the IDs that are missing:")
#         print(list(missing_ids))

#     # 5. Save the result
#     german_metadata.to_csv(output_path, index=False)
#     print(f"File saved to: {output_path}")
    
#     print("\nPreview of extracted data:")
#     print(german_metadata.head())

# if __name__ == "__main__":
#     extract_german_metadata()



#########################################################################################################################################################
#                                                     Extract ids that are for the test set only (language agnostic)

# import pandas as pd
# import os

# def extract_test_set(input_csv_path, output_folder):
#     # 1. Read the original CSV file
#     try:
#         df = pd.read_csv(input_csv_path, sep='\s+')
#         print(f"Successfully loaded {input_csv_path}")
#     except FileNotFoundError:
#         print(f"Error: The file {input_csv_path} was not found.")
#         return

#     # 2. Filter the DataFrame
#     # We look for rows where the column 'Set' is equal to 'test'
#     # Note: We use .strip() just in case there is accidental whitespace in the CSV
#     test_df = df[df['Set'].str.strip() == 'test']

#     # Check if we found any data
#     if test_df.empty:
#         print("Warning: No rows found with Set='test'.")
#     else:
#         print(f"Found {len(test_df)} rows belonging to the test set.")

#     # 3. Prepare the output path
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"Created output folder: {output_folder}")

#     # Define the new filename
#     output_filename = "vox2_meta_test_only.csv"
#     output_path = os.path.join(output_folder, output_filename)

#     # 4. Save the new CSV file
#     # index=False prevents pandas from adding a new 0,1,2... column to the left
#     test_df.to_csv(output_path, index=False)
#     print(f"Successfully saved filtered data to: {output_path}")

# # --- Usage Example ---

# # Define your paths here
# input_path = '/ceph/shared/ALL/datasets/voxceleb2-V2/vox2_meta.csv'  # Path to the file you uploaded
# output_dir = '/ceph/shared/ALL/datasets/voxceleb2-V2'    # Folder where you want the new file

# # Run the function
# extract_test_set(input_path, output_dir)




#########################################################################################################################################################
#                                                     Using the ids for test set, identify the languages in the massive audio clips file


# import pandas as pd
# import os

# def extract_test_audio_clips_debug(test_ids_csv_path, audio_clips_csv_path, output_folder):
#     # --- 1. Load the Test IDs ---
#     print(f"--- DIAGNOSTIC MODE ---")
#     print(f"Loading test IDs from: {test_ids_csv_path}")
    
#     try:
#         df_test_ids = pd.read_csv(test_ids_csv_path)
#         df_test_ids.columns = df_test_ids.columns.str.strip()
        
#         # Determine ID column
#         if 'VoxCeleb2_ID' in df_test_ids.columns:
#             id_col = 'VoxCeleb2_ID'
#         elif 'VoxCeleb2 ID' in df_test_ids.columns:
#             id_col = 'VoxCeleb2 ID'
#         else:
#             print("Error: Could not find ID column.")
#             return

#         # Clean IDs (remove spaces)
#         df_test_ids[id_col] = df_test_ids[id_col].astype(str).str.strip()
#         valid_test_ids = set(df_test_ids[id_col])
        
#         print(f"Loaded {len(valid_test_ids)} unique test IDs.")
#         print(f"Sample Test IDs: {list(valid_test_ids)[:5]}") # PRINT SAMPLE

#     except Exception as e:
#         print(f"Error reading test IDs: {e}")
#         return

#     # --- 2. Process Audio Clips ---
#     print(f"\nProcessing audio clips from: {audio_clips_csv_path}")
#     output_path = os.path.join(output_folder, "audio_clips_test_only.csv")
    
#     chunk_size = 100000
#     total_extracted = 0
#     first_chunk = True
    
#     # We use a flag to print the first chunk's IDs only once
#     debug_printed = False

#     for chunk in pd.read_csv(audio_clips_csv_path, chunksize=chunk_size):
#         chunk.columns = chunk.columns.str.strip()
        
#         if 'speaker_id' not in chunk.columns:
#             print("Error: 'speaker_id' column not found.")
#             return

#         # Clean IDs (remove spaces) - CRITICAL FIX
#         chunk['speaker_id'] = chunk['speaker_id'].astype(str).str.strip()

#         # DIAGNOSTIC: Print IDs from the first chunk to compare
#         if not debug_printed:
#             print(f"Sample Audio Clip IDs: {chunk['speaker_id'].head().tolist()}")
            
#             # Check if ANY ID in this chunk exists in our test set
#             overlap = chunk[chunk['speaker_id'].isin(valid_test_ids)]
#             if overlap.empty:
#                 print("WARNING: The first 100k rows contain NO matches for the test set.")
#                 print("If this file only contains the 'Dev' dataset, this result is expected.")
#             debug_printed = True

#         # Filter
#         filtered_chunk = chunk[chunk['speaker_id'].isin(valid_test_ids)]

#         if not filtered_chunk.empty:
#             mode = 'w' if first_chunk else 'a'
#             header = first_chunk
#             filtered_chunk.to_csv(output_path, mode=mode, index=False, header=header)
#             total_extracted += len(filtered_chunk)
#             first_chunk = False
#             print(f"Found matches! Extracted {total_extracted} rows...", end='\r')

#     print(f"\n\n--- FINISHED ---")
#     print(f"Total rows extracted: {total_extracted}")
#     if total_extracted == 0:
#         print("CONCLUSION: No matches found.")
#         print("Please check the 'Sample Test IDs' vs 'Sample Audio Clip IDs' printed above.")
#         print("If they look different (e.g. 'id00017' vs '00017'), the format is wrong.")
#         print("If they look similar but don't match, your audio_clips file likely does not contain Test data.")

# # --- CONFIGURATION ---
# path_to_test_ids = '/ceph/shared/ALL/datasets/voxceleb2-V2/vox2_meta_test_only.csv'
# path_to_audio_clips = '/ceph/shared/ALL/datasets/voxceleb2-V2/audio_clips_meta_data.csv'
# output_dir = '/ceph/shared/ALL/datasets/voxceleb2-V2/'

# if __name__ == "__main__":
#     extract_test_audio_clips_debug(path_to_test_ids, path_to_audio_clips, output_dir)



#########################################################################################################################################################
#                                                     copy and paste the german dataset to new folders


import pandas as pd
import shutil
import os
from tqdm import tqdm  # Progress bar library

def copy_german_dataset(csv_path, source_aac_root, source_mp4_root, dest_aac_root, dest_mp4_root):
    """
    Copies folders matching IDs from the CSV file to new destination folders.
    """
    
    # 1. Read the CSV to get the list of IDs
    print(f"Reading IDs from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        # Ensure column name matches your file (VoxCeleb2_ID) and clean whitespace
        target_ids = df['VoxCeleb2_ID'].astype(str).str.strip().unique()
        print(f"Found {len(target_ids)} unique speakers to copy.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Create destination directories if they don't exist
    os.makedirs(dest_aac_root, exist_ok=True)
    os.makedirs(dest_mp4_root, exist_ok=True)
    print(f"Destination folders created/verified.")

    # 3. Define the copy logic
    def copy_folder(speaker_id, src_root, dst_root, type_label):
        src_path = os.path.join(src_root, speaker_id)
        dst_path = os.path.join(dst_root, speaker_id)

        if os.path.exists(src_path):
            # Check if already exists to avoid re-copying or errors
            if os.path.exists(dst_path):
                # print(f"Skipping {speaker_id} ({type_label}) - already exists.")
                pass
            else:
                try:
                    shutil.copytree(src_path, dst_path)
                except Exception as e:
                    print(f"Failed to copy {speaker_id} ({type_label}): {e}")
        else:
            # Optional: Print if source is missing (common in some datasets)
            # print(f"Warning: Source folder not found for {speaker_id} in {type_label}")
            pass

    # 4. Iterate and Copy
    print("Starting copy process... this may take a while.")
    
    # We use tqdm for a progress bar
    for speaker_id in tqdm(target_ids, desc="Copying Speakers"):
        # Copy AAC (Audio)
        copy_folder(speaker_id, source_aac_root, dest_aac_root, "AAC")
        
        # Copy MP4 (Video)
        copy_folder(speaker_id, source_mp4_root, dest_mp4_root, "MP4")

    print("\nCopy process finished!")

# --- CONFIGURATION ---

# Path to the CSV containing the German IDs
csv_file_path = '/ceph/shared/ALL/datasets/voxceleb2-V2/vox2_meta_german.csv'

# Source Folders (Where the data is NOW)
src_aac = '/ceph/shared/ALL/datasets/voxceleb2-V2/dev/aac'
src_mp4 = '/ceph/shared/ALL/datasets/voxceleb2-V2/dev/mp4'

# Destination Folders (Where you want the data TO GO)
# CHANGE THESE to your desired output paths
dst_aac = '/ceph/shared/ALL/datasets/voxceleb2-V2/VoxCeleb2-German/dev/aac'
dst_mp4 = '/ceph/shared/ALL/datasets/voxceleb2-V2/VoxCeleb2-German/dev/mp4'


        
copy_german_dataset(csv_file_path, src_aac, src_mp4, dst_aac, dst_mp4)