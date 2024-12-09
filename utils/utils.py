
import os
import time
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from glob import glob

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")
    return path

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "image", "*.*")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.*")))

    test_x = sorted(glob(os.path.join(path, "test", "image", "*.*")))
    test_y = sorted(glob(os.path.join(path, "test", "mask", "*.*")))

    return (train_x, train_y), (test_x, test_y)

def show_image(image_or_path, title):
    """Check if input is a string (image path), """
    if isinstance(image_or_path, str):
        image=cv2.imread(image_or_path, cv2.IMREAD_ANYCOLOR)
    else:
        image=image_or_path
    if len(image.shape()) == 3:
        cv2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        cv2.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_3_image_gray(image1, title1, image2, title2, image3, title3):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 6), sharex=True, sharey=True)

    # Display the first image
    #if len(image1.shape) == 3:  # Color image
     #   ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    #else:  # Grayscale image
    ax1.imshow(image1, cmap='gray')

    ax1.set_title(title1)
    ax1.axis('off')

    # Display the second image
    # if len(image2.shape) == 3:  # Color image
    #     ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    # else:  # Grayscale image
    ax2.imshow(image2, cmap='gray')

    ax2.set_title(title2)
    ax2.axis('off')

    # Display the third image
    # if len(image3.shape) == 3:  # Color image
    #     ax3.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    # else:  # Grayscale image
    ax3.imshow(image3, cmap='gray')

    ax3.set_title(title3)
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

def plot_histogram(image_or_path):
    if isinstance(image_or_path, str):
        image=cv2.imread(image_or_path, cv2.IMREAD_ANYCOLOR)
    else:
        image=image_or_path
    b, g, r = cv2.split(image)
    channels = {'Blue': b, 'Green': g, 'Red': r}
    plot_3_image(b, "Blue", g, "Green", r, "Red")  # Reference images directly
    for key, channel in channels.items():
        print(f'{key} channel data:\n{channel}\n')
        plt.figure(figsize=(10, 5))
        plt.title(f'{key} Color Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.hist(channel.ravel(), bins=256, color=key.lower())  # Add color to histogram
        plt.show()

def adjust_channel_intensity_for_image(image_path, output_image_path, red_factor = 1, green_factor = 1, blue_factor = 1):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

    # Split the image into its channels
    b, g, r = cv2.split(image)

    # Customize
    r = cv2.multiply(r, red_factor)
    g = cv2.multiply(g, green_factor)
    b = cv2.multiply(b, blue_factor)

    # Clip the values to stay in the valid range [0, 255]
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    # Merge the channels back into an image
    modified_image = cv2.merge([b, g, r])
    cv2.imwrite(output_image_path, modified_image)

    return modified_image


def custom_image_intensity(input_folder, output_folder, red_factor=1.0, green_factor=1.0, blue_factor=1.0):
    create_dir(output_folder)
    # Loop through all the files in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)

        # Ensure the file is an image (you can extend this check if needed)
        if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            print(f"Processing {image_name}")
            # Adjust the channel intensity
            modified_image = adjust_channel_intensity_for_image(image_path, red_factor, green_factor, blue_factor)

            # Save the image to the output folder, keeping the original name
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, modified_image)
            print(f"Saved modified image to {output_image_path}")

def apply_threshold(image_or_path, threshold_value):
    if isinstance(image_or_path, str):
        image=cv2.imread(image_or_path, cv2.IMREAD_ANYCOLOR)
    else:
        image=image_or_path
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    show_image(gray, 'Original Grayscale')
    show_image(threshold, 'Threshold Image')
    print(f"Binary Image Array:\n{threshold}")
    return threshold


def create_dir_structure(base_path):
    """Create a directory structure based on the base path."""
    # Define the subdirectories you want to create
    subdirs = [
        'train/image',
        'train/mask',
        'test/image',
        'test/mask'
    ]

    # Create each subdirectory
    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def concatenate_image_and_mask(image_path, mask_path, output_path):
    name = os.path.basename(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # (H, W, 3)
    # Read the grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
    # Ensure that the image and mask have the same dimensions
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    # Normalize the image and mask to [0, 1]
    image = image / 255.0
    mask = mask / 255.0

    # Expand the mask dimensions to match the image channels
    mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)

    # Concatenate the mask with the image
    concatenated = np.concatenate((image, mask), axis=-1)  # (H, W, 4)

    # Save the concatenated image
    output_path = os.path.join(output_path, name)
    concatenated = (concatenated * 255).astype(np.uint8)  # Convert back to [0, 255] range
    cv2.imwrite(output_path, concatenated)
    print(f'Concatenate sucessfully image {name}.')

def add_label(image, label, position=(10, 30), font_scale=1, color=(255, 255, 255), thickness=2):
    """Adds a text label to the image."""
    if len(image.shape) == 2:  # Convert grayscale to BGR if needed
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image
