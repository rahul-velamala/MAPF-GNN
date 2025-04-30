import scipy.io

# Load the .mat file
data = scipy.io.loadmat('dataset/DataSource_DMap_FixedComR/EffectiveDensity/Training/map20x20_density_p1/10_Agent/valid/valid_IDMap00800_IDCase00000_MP26.mat')

# Open a txt file to save the contents
with open('output.txt', 'w') as f:
    for key, value in data.items():
        # Skip MATLAB metadata fields
        if key.startswith('__'):
            continue
        f.write(f"Variable Name: {key}\n")
        f.write(f"Value:\n{value}\n\n")