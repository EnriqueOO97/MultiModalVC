import os

def update_paths(file_path, old_prefix, new_prefix):
    """
    Reads a TSV file, replaces the old path prefix with the new one,
    and overwrites the file with the changes.
    """
    print(f"Processing {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Open the file for writing (overwriting)
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                new_line = line.replace(old_prefix, new_prefix)
                f.write(new_line)
                
        print(f"Successfully updated {file_path}")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

def main():
    # --- CONFIGURATION ---
    
    # 1. Directory containing the TSV files
    tsv_folder_path = '/data/fs201163/eo49197/MultiModalVC/manifest/germanManifest/' 

    # 2. string segment to find
    old_root = "/ceph/shared/ALL/datasets/"
    
    # 3. string segment to replace it with
    new_root = "/data/fs201163/eo49197/"
    
    # ---------------------

    print("Starting path adjustment...")
    print(f"Target Folder: {tsv_folder_path}")
    print(f"Replacing '{old_root}' with '{new_root}'")
    print("-" * 30)

    # Target only the three MOS manifest files
    tsv_files = [
        os.path.join(tsv_folder_path, "trainMOS.tsv"),
        os.path.join(tsv_folder_path, "validMOS.tsv"),
        os.path.join(tsv_folder_path, "testMOS.tsv"),
    ]

    for tsv in tsv_files:
        update_paths(tsv, old_root, new_root)

    print("-" * 30)
    print("Done.")

if __name__ == "__main__":
    main()