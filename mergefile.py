import numpy as np
import os
from pathlib import Path

def merge_npz_files(base_dir, output_file, folder_range_start=0, folder_range_end=100):
    """
    Merge demonstrations.npz files from multiple folders into a single npz file.
    
    Args:
        base_dir: Path to the dispatch_datasets directory
        output_file: Output npz file name
        folder_range_start: Starting folder number (inclusive)
        folder_range_end: Ending folder number (inclusive)
    """
    
    all_data = {}
    
    # First, load one file to understand the structure
    first_file = Path(base_dir) / str(folder_range_start) / "demonstrations.npz"
    sample_data = np.load(first_file)
    print(f"Sample file keys: {list(sample_data.keys())}")
    
    # Initialize lists for each key to store concatenated data
    for key in sample_data.keys():
        all_data[key] = []
    
    # Load and concatenate all files
    for folder_num in range(folder_range_start, folder_range_end + 1):
        npz_path = Path(base_dir) / str(folder_num) / "demonstrations.npz"
        
        if not npz_path.exists():
            print(f"Warning: {npz_path} does not exist, skipping...")
            continue
        
        try:
            data = np.load(npz_path)
            
            for key in data.keys():
                all_data[key].append(data[key])
            
            print(f"Loaded folder {folder_num}")
            
        except Exception as e:
            print(f"Error loading folder {folder_num}: {e}")
            continue
    
    # Concatenate arrays for each key
    merged_data = {}
    for key, arrays in all_data.items():
        if len(arrays) > 0:
            # Check if it's a 0-d array (scalar) or 1-d array
            if arrays[0].ndim == 0:
                # For scalar values, just keep the first one or create an object array
                merged_data[key] = np.array(arrays, dtype=object)
                print(f"Merged {key} (scalar): shape = {merged_data[key].shape}")
            else:
                # For regular arrays, concatenate along axis 0
                merged_data[key] = np.concatenate(arrays, axis=0)
                print(f"Merged {key}: shape = {merged_data[key].shape}")
        else:
            print(f"Warning: No data found for key {key}")
    
    # Save the merged data
    output_path = Path(base_dir) / output_file
    np.savez_compressed(output_path, **merged_data)
    print(f"\nMerged file saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")


if __name__ == "__main__":
    base_dir = r"c:\Capstone\benchmark\-Optimize-Disaster-Recovery\experiment_results\dispatch_datasets"
    output_file = "1_100.npz"
    
    merge_npz_files(base_dir, output_file, folder_range_start=0, folder_range_end=100)
