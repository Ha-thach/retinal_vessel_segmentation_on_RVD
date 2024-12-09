import os
import numpy as np
import cv2
from tqdm import tqdm
import yaml

from utils.utils import create_dir, seeding, load_data, add_label



def calculate_positive_pixels(image):
    """Calculate the number of positive pixels in the image."""
    return np.sum(image / 255 > 0)

if __name__ == "__main__":
    """ Load configuration from YAML """
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    """ Seeding """
    seeding(config['seed'])

    """ Folders to predicted mask images """
    result_path_1 = config['result_path_1'].strip()
    result_path_2 = config['result_path_2'].strip()
    result_path_3 = config['result_path_3'].strip()
    concatenate_folder = config['save_concatenate_folder']
    create_dir(concatenate_folder)

    """ Load dataset """
    data_path = config['data']
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    data_str = f"Dataset Size:\nTest: {len(test_x)}"
    print(data_str)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Read and preprocess input image """
        original_image_path = x
        name = os.path.splitext(os.path.basename(x))[0]
        print(name)

        # Read the color image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Unable to read image at {x}")
            continue

        """ Read and preprocess ground truth mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Unable to read mask at {y}")
            continue

        """ Load the predicted masks from U-Net """
        image_1_path = os.path.join(result_path_1, f"{name}.png")
        image_2_path = os.path.join(result_path_2, f"{i}.png")
        image_3_path = os.path.join(result_path_3, f"{i}.png")

        # Read the masks
        image_1 = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)
        image_2 = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)
        image_3 = cv2.imread(image_3_path, cv2.IMREAD_GRAYSCALE)

        # Check if any image is None
        if image_1 is None:
            print(f"Error: Unable to read image at {image_1_path}")
            continue
        if image_2 is None:
            print(f"Error: Unable to read image at {image_2_path}")
            continue
        if image_3 is None:
            print(f"Error: Unable to read image at {image_3_path}")
            continue

        """ Add labels to the images """
        image_labeled = add_label(image, "Image")
        mask_labeled = add_label(mask, "Original Mask")
        first_labeled = add_label(image_1, "UNet")
        second_labeled = add_label(image_2, "SAM")
        third_labeled = add_label(image_3, "SAM-UNet Decoder")

        """ Concatenate images for comparison """
        first_row = np.concatenate((image_labeled, mask_labeled, first_labeled, second_labeled, third_labeled), axis=1)

        """ Save the concatenated comparison image """
        comparison_image_path = os.path.join(concatenate_folder, f"{name}.png")
        print(f"Saving image: {comparison_image_path}")
        cv2.imwrite(comparison_image_path, first_row)
