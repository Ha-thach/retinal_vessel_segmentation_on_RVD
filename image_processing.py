import os
import cv2

# Function to process images
def process_image(input_path, output_path):
    # Read the color image
    image = cv2.imread(input_path)

    # Split the image into its RGB channels
    (b, g, r) = cv2.split(image)

    #  create a green channel image
    image = cv2.imwrite(output_path, r)
    return image

# Path to the directory containing the dataset
dataset_dir = "/data/nhthach/project/RETINAL/DATA/subset_train_test/train/image"

# Path to the directory where you want to save the new dataset
new_dataset_dir = "/data/nhthach/project/RETINAL/DATA/subset_train_test/train_red"

# Iterate through all files in the dataset directory
for filename in os.listdir(dataset_dir):
    # Check if the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct the input and output paths
        input_path = os.path.join(dataset_dir, filename)
        output_path = os.path.join(new_dataset_dir, filename)

        # Process the image and save it to the new dataset
        process_image(input_path, output_path)




