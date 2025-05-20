import os

directory = "dataset/map10x10_r5_o20_p5/val"

# Count only directories (not files)
folder_count = sum(
    os.path.isdir(os.path.join(directory, item))
    for item in os.listdir(directory)
)

print(f"Number of folders in '{directory}': {folder_count}")
