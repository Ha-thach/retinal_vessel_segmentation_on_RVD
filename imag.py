import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def show_image(image, title):
    if  len(image.shape)==3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image_from_path(image_path, title):
    image=cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    if  len(image.shape)==3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_comparison(image1, title1, image2, title2, image3, title3):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 6), sharex=True, sharey=True)

    # Display the first image
    if len(image1.shape) == 3:  # Color image
        ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        ax1.imshow(image1, cmap='gray')

    ax1.set_title(title1)
    ax1.axis('off')

    # Display the second image
    if len(image2.shape) == 3:  # Color image
        ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        ax2.imshow(image2, cmap='gray')

    ax2.set_title(title2)
    ax2.axis('off')
# Display the third image
    if len(image3.shape) == 3:  # Color image
        ax3.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        ax3.imshow(image3, cmap='gray')

    ax3.set_title(title3)
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

def plot_histogram(image):  # Plot histograms for each channel
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    b, g, r = cv2.split(image)
    channels = {'Blue': b, 'Green': g, 'Red': r}
    plot_comparison(b, "Blue", g, "Green", r, "Red")
    for key in channels:
        print(channels[key])
        print(key)
        plt.figure(figsize=(10, 5))
        plt.title(f'{key} Color Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.hist(channels[key].ravel(), bins=256)
        plt.show()


def apply_threshold(image_path, threshold_value):
    image=cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,binary_image=cv2.threshold(gray, threshold_value,255, cv2.THRESH_BINARY)
    show_image(gray,'orginal1')
    show_image(binary_image, 'threshold')
    print(binary_image)


def vein_threshold_mask(input_folder, output_folder, threshold_value):
    #Create a output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

                # Apply threshold to the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                # Save the thresholded image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary_image)
        print(f"Processed {filename} and saved to {output_path}")


#image=show_image_from_path(image_path, "original")
#plot_histogram(image_path)
#apply_threshold(image_path, 120)
#input_folder = '/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/train/Artery_vein_mask'
#output_folder= '/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/train/vein_mask_only'
#vein_threshold_mask(input_folder, output_folder, 120)


def crop_image_with_background(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    show_image(image, 'original')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    # Find contours in the binary image
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the object to keep
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Extract the object using the mask
    object_only = cv2.bitwise_and(image, image, mask=mask)

    # Find bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_image = object_only[y:y+h, x:x+w]

    show_image(cropped_image, "cropped")
    #plot_comparison(image, "original", contours, "contour", cropped_image, "crop")

# Example usage
image_path = '/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/train/IMAGE/0001_54.png'
crop_image_with_background(image_path)


