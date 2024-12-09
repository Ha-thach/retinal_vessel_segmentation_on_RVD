import os
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomResizedCrop
from utils.utils import load_data, create_dir, create_dir_structure


import cv2
import numpy as np

# Use the calculated mean and std for normalization
mean = np.array([0.3807, 0.1883, 0.0998])
std = np.array([0.2927, 0.1607, 0.1053])

def apply_equalization(image):
    """ Apply histogram equalization to each channel of the image """
    # Split the image into R, G, and B channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply histogram equalization to each channel
    r_equalized = cv2.equalizeHist(r_channel)
    g_equalized = cv2.equalizeHist(g_channel)
    b_equalized = cv2.equalizeHist(b_channel)

    # Merge the equalized channels back
    equalized_image = cv2.merge([b_equalized, g_equalized, r_equalized])

    return equalized_image

def preprocess_image(image, size=(512, 512), equalize=False, colornormalization=False):
    """ Basic preprocessing steps: resizing, optional equalization, and normalization """
    # Resize the image
    image = cv2.resize(image, size)

    # Optional histogram equalization
    if equalize:
        image = apply_equalization(image)

    # Convert image to float and normalize to [0, 1]
    image = image / 255.0

    # Normalize image using the calculated mean and std
    if colornormalization:
        image = (image - mean) / std  # Channel-wise normalization

    return image



def preprocess_mask(mask, size=(512, 512)):
    """ Resize the mask """
    mask = cv2.resize(mask, size)
    return mask

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        """ Preprocess image and mask """
        x = preprocess_image(x, size, equalize=False, colornormalization=False)
        y = preprocess_mask(y, size)

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug= RandomResizedCrop(height=512, width=512, scale=(0.1, 0.5), ratio=(1, 1), p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            X = [x, x1, x2, x3, x4]
            Y = [y, y1, y2, y3, y4]


        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, (i * 255).astype(np.uint8))  # Convert back to [0, 255]
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "/data/nhthach/project/DATA/Retinal_Fractal/Original"

    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the preprocessed data """
    path = "/data/nhthach/project/DATA/Retinal_Fractal/Original-1"
    create_dir(path)
    create_dir_structure(path)
    save_path_for_train = os.path.join(path, "train")
    save_path_for_test =  os.path.join(path, "test")
    """ Data preprocessing """
    augment_data(train_x, train_y, save_path_for_train, augment=False)
    augment_data(test_x, test_y, save_path_for_test, augment=False)



