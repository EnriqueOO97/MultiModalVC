def calculate_total_duration(tsv_file_path, sample_rate=16000):
    total_samples = 0
    
    try:
        with open(tsv_file_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip the first line (it contains the root path "/")
            data_lines = lines[1:]
            
            print(f"Processing {len(data_lines)} lines...")
            
            for line in data_lines:
                parts = line.strip().split('\t')
                
                # We expect at least 5 columns. 
                # The structure seems to be: ID | VideoPath | AudioPath | VideoFrames | AudioSamples | Score
                if len(parts) >= 5:
                    try:
                        # Column index 4 is the audio sample count (e.g., 67584)
                        n_samples = int(parts[4])
                        total_samples += n_samples
                    except ValueError:
                        continue

        # Calculate duration
        total_seconds = total_samples / sample_rate
        total_hours = total_seconds / 3600
        
        print(f"Total samples: {total_samples}")
        print(f"Total seconds: {total_seconds:.2f}")
        print(f"Total hours: {total_hours:.2f}")
        
    except FileNotFoundError:
        print(f"Error: The file '{tsv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
calculate_total_duration('/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/test.tsv')