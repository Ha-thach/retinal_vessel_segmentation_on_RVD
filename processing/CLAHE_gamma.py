import os
from tqdm import tqdm
import cv2
import numpy as np
import imageio
from utils.utils import load_data, create_dir, create_dir_structure

def clahe_equalized(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def sharpen_image(img):
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply the sharpening kernel to the image
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def apply_gaussian_filter(img, kernel_size=(5, 5), sigma=1.0):
    """
    Applies a Gaussian filter to the image.
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)

def do_process(images, masks, save_path):
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]
        print(f"Processing image: {name}")

        """ Reading image and mask """
        img = cv2.imread(x, cv2.IMREAD_COLOR)  # Read the image in color
        y = imageio.mimread(y)[0]  # Read the mask

        """ Split image into channels """
        b_channel, g_channel, r_channel = cv2.split(img)

        """ Preprocess each channel """
        r_processed = adjust_gamma(clahe_equalized(apply_gaussian_filter(r_channel)))
        g_processed = adjust_gamma(clahe_equalized(apply_gaussian_filter(g_channel)))
        b_processed = adjust_gamma(clahe_equalized(apply_gaussian_filter(b_channel)))

        """ Merge the processed channels back """
        processed_image = cv2.merge((b_processed, g_processed, r_processed))

        """ Apply Gaussian filter """
        #processed_image = apply_gaussian_filter(processed_image)

        """ Apply sharpening filter to the processed image """
        processed_image = sharpen_image(processed_image)

        name = f"{name}.png"
        image_path = os.path.join(save_path, "image", name)
        mask_path = os.path.join(save_path, "mask", name)

        # Ensure the images are in the proper range before saving
        print(f'Saving processed image {name} to {image_path}')
        cv2.imwrite(image_path, processed_image.astype(np.uint8))
        cv2.imwrite(mask_path, y)  # Save the original mask for all images

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "/data/nhthach/project/DATA/Retinal_Fractal/Original-1"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the preprocessed data """
    path = "/data/nhthach/project/DATA/Retinal_Fractal/Original_gaussian_clahe_gamma_sharpen_2"
    create_dir(path)
    create_dir_structure(path)
    save_path_for_train = os.path.join(path, "train")
    save_path_for_test = os.path.join(path, "test")

    """ Data preprocessing """
    do_process(train_x, train_y, save_path_for_train)
    do_process(test_x, test_y, save_path_for_test)
