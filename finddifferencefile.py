import os
import shutil

# Define the paths to folder A and folder B
folder_A_path = "/data/nhthach/project/RETINAL/DATA/image_mask_combined/IMAGE"
folder_B_path = "/data/nhthach/project/RETINAL/DATA/image_mask_combined/Artery_Vein_Mask"

# Define the path to the new folder
new_folder_path = "/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein"

# Create the new folder if it doesn't exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# Create folders A and B inside the new folder
folder_A_new_path = os.path.join(new_folder_path, "IMAGE")
folder_B_new_path = os.path.join(new_folder_path, "Artery_vein_mask")
os.makedirs(folder_A_new_path)
os.makedirs(folder_B_new_path)

# Get the list of files in folder A
files_in_A = os.listdir(folder_A_path)

# Get the list of files in folder B
files_in_B = os.listdir(folder_B_path)

# Find files with the same name in both folders
common_files = set(files_in_A).intersection(files_in_B)

# Copy common files to the new folder, preserving folder structure
for file_name in common_files:
    source_A = os.path.join(folder_A_path, file_name)
    source_B = os.path.join(folder_B_path, file_name)
    destination_A = os.path.join(folder_A_new_path, file_name)
    destination_B = os.path.join(folder_B_new_path, file_name)
    # Copy the file only if it exists in both folders
    if os.path.exists(source_A) and os.path.exists(source_B):
        shutil.copyfile(source_A, destination_A)
        shutil.copyfile(source_B, destination_B)

print("Common files copied to the new folder successfully.")
