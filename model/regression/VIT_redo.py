import os

# Directory containing the files
directory = "/data/nhthach/project/results/Original-aug_gaussian_clahe_gamma_sharpen/Unet-lre-4_Adam1_0.3_ep40_CosineAnnealingL/mask_result_on_fractal_test_2_dataset"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the filename ends with "_0.png"
    if filename.endswith('_0.png'):
        # Create the new filename by removing the "_0" part
        new_filename = filename.replace('_0.png', '.png')
        # Construct full paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)

print("Renaming completed!")
