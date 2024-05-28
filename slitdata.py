import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to your image and mask folders
image_folder = "/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/IMAGE"
mask_folder = "/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/Artery_vein_mask"

# List all image and mask filenames
image_files = os.listdir(image_folder)
mask_files = os.listdir(mask_folder)

# Split filenames into train and test sets for both image and mask datasets
image_train, image_test = train_test_split(image_files, test_size=0.3, random_state=42)
mask_train, mask_test = train_test_split(mask_files, test_size=0.3, random_state=42)

# Create directories for train and test sets for both image and mask datasets
train_dir_image = "/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/train/IMAGE"
test_dir_image = "/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/test/IMAGE"
train_dir_mask = "/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/train/Artery_vein_mask"
test_dir_mask = "/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/test/Artery_vein_mask"

os.makedirs(train_dir_image, exist_ok=True)
os.makedirs(test_dir_image, exist_ok=True)
os.makedirs(train_dir_mask, exist_ok=True)
os.makedirs(test_dir_mask, exist_ok=True)

# Copy images and masks to train and test directories
for image_file, mask_file in zip(image_train, mask_train):
    shutil.copy(os.path.join(image_folder, image_file), os.path.join(train_dir_image, image_file))
    shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(train_dir_mask, mask_file))

for image_file, mask_file in zip(image_test, mask_test):
    shutil.copy(os.path.join(image_folder, image_file), os.path.join(test_dir_image, image_file))
    shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(test_dir_mask, mask_file))

print("Dataset split into train and test sets successfully.")
