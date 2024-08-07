import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, \
    CoarseDropout


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


def augment_data(image_paths, mask_paths, save_path, augment=True):
    H = 512
    W = 512

    for idx, (image_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths)):
        """ Extracting names """
        name = image_path.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = imageio.imread(mask_path)

        if augment:
            aug_list =[
                HorizontalFlip(p=1.0),
                VerticalFlip(p=1.0),
                ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(p=1),
                OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ]

            augmented_images = [image]
            augmented_masks = [mask]
            for aug in aug_list:
                augmented = aug(image=image, mask=mask)
                augmented_images.append(augmented["image"])
                augmented_masks.append(augmented["mask"])

        else:
            augmented_images = [image]
            augmented_masks = [mask]

        for index, (img, msk) in enumerate(zip(augmented_images, augmented_masks)):
            img = cv2.resize(img, (W, H))
            msk = cv2.resize(msk, (W, H))

            tmp_image_name = f"{name}_{index}.png" if len(augmented_images) > 1 else f"{name}.png"
            tmp_mask_name = f"{name}_{index}.png" if len(augmented_images) > 1 else f"{name}.png"

            image_path = os.path.join(save_path, "Image", tmp_image_name)
            mask_path = os.path.join(save_path, "VeinMask", tmp_mask_name)

            cv2.imwrite(image_path, img)
            cv2.imwrite(mask_path, msk)


def main():

    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop)vein"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    train_save_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop&Augted)vein/Train"
    test_save_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop&Augted)vein/Test"

    create_dir(os.path.join(train_save_path, "Image"))
    create_dir(os.path.join(train_save_path, "VeinMask"))
    create_dir(os.path.join(test_save_path, "Image"))
    create_dir(os.path.join(test_save_path, "VeinMask"))

    """ Process and save augmented data """
    augment_data(train_x, train_y, train_save_path, augment=True)
    augment_data(test_x, test_y, test_save_path, augment=False)

if __name__ == "__main__":
    main()