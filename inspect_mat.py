import scipy.io
import numpy as np
from pathlib import Path

# --- SET PATH TO ONE .mat FILE ---
mat_file_path = Path('/scratch/rahul/v1/project/MAPF-GNN-ADC/dataset/DataSource_DMap_FixedComR/EffectiveDensity/Training/map20x20_density_p1/10_Agent/train/train_IDMap00000_IDCase00000_MP29.mat') # Example path

if not mat_file_path.is_file():
    print(f"Error: File not found at {mat_file_path}")
else:
    try:
        print(f"Loading: {mat_file_path}")
        data = scipy.io.loadmat(mat_file_path)
        print("\nKeys in the .mat file:")
        print(list(data.keys()))

        print("\nDetails for potentially relevant keys:")
        for key in data:
            if not key.startswith('__'): # Ignore metadata keys
                item = data[key]
                print(f"  Key: '{key}'")
                if isinstance(item, np.ndarray):
                    print(f"    Type: NumPy array")
                    print(f"    Shape: {item.shape}")
                    print(f"    Dtype: {item.dtype}")
                else:
                    print(f"    Type: {type(item)}")
                    # print(f"    Value: {item}") # Uncomment carefully for small items

    except Exception as e:
        print(f"An error occurred: {e}")