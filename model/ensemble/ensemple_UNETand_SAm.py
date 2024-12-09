import os

import numpy as np
import cv2
from tqdm import tqdm
import yaml
import csv

from utils.utils import create_dir, seeding, load_data


def add_label(image, label, position=(10, 30), font_scale=1, color=(255), thickness=2):
    """Adds a text label to the image."""
    labeled_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored text
    cv2.putText(labeled_image, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, lineType=cv2.LINE_AA)
    return labeled_image
def calculate_positive_pixels(image):
    """Calculate the number of positive pixels in the image."""
    return np.sum(image/255 > 0)

if __name__ == "__main__":
    """ Load configuration from YAML """
    with open("../../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    """ Seeding """
    seeding(config['seed'])

    """ Folders to predicted mask images """
    unet_mask_result_path = config['unet_mask_result_path']
    sam_mask_result_path = config['sam_mask_result_path']
    concatenate_folder = config['concatenate_folder']
    create_dir(concatenate_folder)

    """ Load dataset """
    data_path = config['data_path']
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    data_str = f"Dataset Size:\nTest: {len(test_x)}"
    print(data_str)
    results = []
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Read and preprocess input image """
        original_image_path = x
        name = os.path.splitext(os.path.basename(x))[0]
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # (512, 512, 3)

        """ Read and preprocess ground truth mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE).astype(np.int16)  # (512, 512)

        """ Load the predicted masks from U-Net """
        unet_path = os.path.join(unet_mask_result_path, f"{name}.png")

        # SAM mask path is sequential, starting from 0 to len(test_x) - 1
        sam_path = os.path.join(sam_mask_result_path, f"{i}.png")

        # Check if the mask files exist
        if not os.path.exists(unet_path):
            print(f"U-Net mask not found for {name}. Skipping...")
            continue
        if not os.path.exists(sam_path):
            print(f"SAM mask not found for index {i}. Skipping...")
            continue

        # Read the masks
        unet = cv2.imread(unet_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)  # (512, 512)
        sam = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE).astype(np.int16) # (512, 512)
        # print(mask.shape)
        # print(unet.shape)
        # print(sam.shape)
        # np.set_printoptions(threshold=sys.maxsize)
        # temp = sam.astype(np.int16) - mask.astype(np.int16)
        # # print(temp[temp < 0])
        #
        # show_image(np.clip(temp, 0, 255), 'test')
        mask_minus_sam = np.clip(mask - sam, 0, 255).astype(np.uint8)
        mask_minus_unet = np.clip(mask - unet, 0, 255).astype(np.uint8)
        unet_minus_mask = np.clip(unet - mask, 0, 255).astype(np.uint8)
        unet_minus_sam = np.clip(unet - sam,0, 255).astype(np.uint8)
        sam_minus_mask = np.clip(sam - mask, 0, 512).astype(np.uint8)
        sam_minus_unet = np.clip(sam - unet, 0, 255).astype(np.uint8)
        mask_sam_unet = np.clip(mask - sam_minus_unet, 0, 255).astype(np.uint8)
        sam_unet_mask = np.clip(sam_minus_unet - mask, 0, 255).astype(np.uint8)
        """ Calculate positive pixel sums for the difference images """
        positive_pixels_sam_minus_unet = calculate_positive_pixels(sam_minus_unet)
        positive_pixels_sam_unet_mask = calculate_positive_pixels(sam_unet_mask)

        # Append the results to the list
        results.append([name, positive_pixels_sam_minus_unet, positive_pixels_sam_unet_mask])

        """ Add labels to the images """
        mask_labeled = add_label(mask.astype(np.uint8), "Original Mask")
        unet_labeled = add_label(unet.astype(np.uint8), "UNet")
        sam_labeled = add_label(sam.astype(np.uint8), "SAMUnetD")
        mask_minus_unet = add_label(mask_minus_unet, "Mask-UNet")
        unet_minus_mask = add_label(unet_minus_mask, "UNet-Mask")
        unet_minus_sam = add_label(unet_minus_sam, "UNet-SAMUnetD")
        mask_minus_sam = add_label(mask_minus_sam, "Mask-SAMUnetD")
        sam_minus_mask = add_label(sam_minus_mask, "SAMUnetD-Mask")
        sam_minus_unet = add_label(sam_minus_unet, "SAMUnetD-UNet")
        mask_sam_unet = add_label(mask_sam_unet, "Mask-(SAMUnetD-UNet)")
        sam_unet_mask = add_label(sam_unet_mask, "(SAMUnetD-UNet)-Mask")

        # Compute U-Net - mask difference
        """ Concatenate images for comparison """
        # First row: Original Mask repeated 4 times
        first_row = np.concatenate((mask_labeled, mask_labeled, mask_sam_unet, sam_unet_mask), axis=1)

        # Second row: U-Net Mask, Original Mask - U-Net, U-Net - SAM Diff, Mask - SAM
        second_row = np.concatenate((unet_labeled, mask_minus_unet, unet_minus_mask, unet_minus_sam), axis=1)

        # Third row: SAM Mask, Original Mask - SAM, SAM - U-Net Diff, U-Net - SAM Diff
        third_row = np.concatenate((sam_labeled, mask_minus_sam, sam_minus_mask, sam_minus_unet), axis=1)

        # Concatenate the rows vertically to form the final image
        final = np.concatenate((first_row, second_row, third_row), axis=0)

        # Save the concatenated comparison image
        comparison_image_path = os.path.join(concatenate_folder, f"{name}.png")
        print(f"Saving image: {comparison_image_path}")
        cv2.imwrite(comparison_image_path, final)


    csv_file_path = os.path.join(concatenate_folder, "positive_pixel_counts.csv")
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Positive Pixels (SAM - UNet)", "Positive Pixels ((SAM - UNet) - Mask)"])
        writer.writerows(results)
