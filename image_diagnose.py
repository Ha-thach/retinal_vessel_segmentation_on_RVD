




import os
import cv2
import numpy as np


# Function to process images
def process_image(input_path, output_path):
    # Read the color image
    image=cv2.imread(input_path)

    #print(image.shape)

    # Split the image into its RGB channels
    (b, g, r) = cv2.split(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blank = np.zeros_like(b)
    #print("shape of blank", blank.shape)
    blue = cv2.merge([b, blank, blank])
    blue=cv2.cvtColor(blue,cv2.COLOR_BGR2GRAY)
    red = cv2.merge([blank, blank, r])
    red = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    green= cv2.merge([blank, g, blank])
    green = cv2.cvtColor(green,cv2.COLOR_BGR2GRAY)
    print(image.shape)
    concate_image=np.concatenate([image, blue, green, red], axis=1)

    #  create a one channel image
    #cv2.imwrite(output_path, concate_image)
    #ret, thresh1 = cv2.threshold(image,150,200,cv2.THRESH_BINARY)
    cv2.imwrite(output_path, concate_image)
# Path to the directory containing the dataset
dataset_dir = "/data/nhthach/project/RETINAL/DATA/subset_train_test/train/image"

# Path to the directory where you want to save the new dataset
new_dataset_dir = "/data/nhthach/project/RETINAL/DATA/subset_train_test/compare"

# Iterate through all files in the dataset directory
for filename in os.listdir(dataset_dir):
    # Check if the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct the input and output paths
        input_path = os.path.join(dataset_dir, filename)
        output_path = os.path.join(new_dataset_dir, filename)

        # Process the image and save it to the new dataset
        process_image(input_path, output_path)
