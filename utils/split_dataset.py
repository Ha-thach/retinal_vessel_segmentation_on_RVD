import os
import shutil
import random

def make_subset(full_dataset_dir, subset_dataset_dir, percentage):
    # Create directories for the subset if they don't exist
    for phase in ['train', 'test']:
        for subfolder in ['image', 'mask']:
            os.makedirs(os.path.join(subset_dataset_dir, phase, subfolder), exist_ok=True)

    # Define helper function to copy files
    def copy_files(src_dir, dest_dir, file_list):
        for file_name in file_list:
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy(src_path, dest_path)

    # Iterate over 'train' and 'test' folders
    for phase in ['train', 'test']:
        image_dir = os.path.join(full_dataset_dir, phase, 'image')
        mask_dir = os.path.join(full_dataset_dir, phase, 'mask')

        # Get all file names in the image folder (assuming both 'image' and 'mask' folders have the same file names)
        file_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        # Calculate the number of files to copy based on the percentage
        num_files_to_copy = int(len(file_names) * (percentage / 100))

        # Randomly select a subset of files
        subset_file_names = random.sample(file_names, num_files_to_copy)

        # Copy selected files to the subset directory for both images and masks
        copy_files(image_dir, os.path.join(subset_dataset_dir, phase, 'image'), subset_file_names)
        copy_files(mask_dir, os.path.join(subset_dataset_dir, phase, 'mask'), subset_file_names)

# Example usage
full_dataset_dir = '/data/nhthach/project/DATA/RETINAL/Binary/train_test_original'
subset_dataset_dir = '/data/nhthach/project/DATA/RETINAL/Binary/binary_0.15'
percentage = 15  # Define the percentage of data you want in the subset

make_subset(full_dataset_dir, subset_dataset_dir, percentage)
