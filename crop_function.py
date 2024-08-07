
import cv2
import numpy as np
from glob import glob
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm


def crop_image_with_background(image_paths, mask_paths, save_image_path, save_mask_path):
    H = 512
    W = 512

    for idx, (image_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths)):
        """ Extracting names """
        name = image_path.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        vein_mask = imageio.imread(mask_path)


        # Ensure the image is loaded
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' not found.")

        # Apply Gaussian blur to the grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply binary thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours are found
        if not contours:
            raise ValueError("No contours found in the image.")

        # Assume the largest contour is the object to keep
        largest_contour = max(contours, key=cv2.contourArea)

        # Find bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Create a mask with a rectangular bounding box
        mask_for_crop = np.zeros_like(gray)
        cv2.rectangle(mask_for_crop, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

        # Apply the contour_mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=mask_for_crop)

        # Crop the image to the bounding box
        cropped_image = masked_image[y:y + h, x:x + w]

        # Apply the contour_mask to the original mask
        masked_mask = cv2.bitwise_and(vein_mask, vein_mask, mask=mask_for_crop)

        # Crop the mask to the bounding box
        cropped_mask = masked_mask[y:y + h, x:x + w]

        save_image_file_path = os.path.join(save_image_path, f"{name}.png")
        save_mask_file_path = os.path.join(save_mask_path, f"{name}.png")

        cv2.imwrite(save_image_file_path, cropped_image)
        cv2.imwrite(save_mask_file_path, cropped_mask)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    """ X = Image and Y = VeinMask """
    train_x = sorted(glob(os.path.join(path, "Train", "Image", "*.png")))
    train_y = sorted(glob(os.path.join(path, "Train", "VeinMask", "*.png")))
    test_x = sorted(glob(os.path.join(path, "Test", "Image", "*.png")))
    test_y = sorted(glob(os.path.join(path, "Test", "VeinMask", "*.png")))
    return (train_x, train_y), (test_x, test_y)

def main():

    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(NoAug)vein"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    save_train_x_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop)vein/Train/Image"
    save_train_y_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop)vein/Train/VeinMask"
    save_test_x_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop)vein/Test/Image"
    save_test_y_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop)vein/Test/VeinMask"

    create_dir(save_train_x_path)
    create_dir(save_train_y_path)
    create_dir(save_test_x_path)
    create_dir(save_test_y_path)


    """ Processing and saving cropped images """
    crop_image_with_background(train_x, train_y, save_train_x_path, save_train_y_path)
    crop_image_with_background(test_x, test_y, save_test_x_path, save_test_y_path)

if __name__ == "__main__":
    main()

