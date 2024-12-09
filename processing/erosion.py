import os
import cv2


def apply_erode_to_color_image(image, kernel_size=(5, 5)):
    """Apply erosion to each channel of the color image."""
    # Split the image into B, G, R channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Define the kernel for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply erosion to each channel
    b_eroded = cv2.erode(b_channel, kernel, iterations=1)
    g_eroded = cv2.erode(g_channel, kernel, iterations=1)
    r_eroded = cv2.erode(r_channel, kernel, iterations=1)

    # Merge the eroded channels back into a color image
    eroded_image = cv2.merge([b_eroded, g_eroded, r_eroded])

    return eroded_image


def process_images(input_folder, output_folder, apply_erode=True, kernel_size=(5, 5)):
    """Process and apply erosion to all images in train and test folders."""
    for subdir in ['train/image', 'test/image']:
        input_subdir = os.path.join(input_folder, subdir)
        output_subdir = os.path.join(output_folder, subdir)

        # Ensure the output directory exists
        os.makedirs(output_subdir, exist_ok=True)

        # Process each image in the subdirectory
        for image_name in os.listdir(input_subdir):
            image_path = os.path.join(input_subdir, image_name)

            # Check if it's an image file
            if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                print(f"Processing {image_name}")

                # Read the image
                image = cv2.imread(image_path)

                # Apply erosion if required
                if apply_erode:
                    image = apply_erode_to_color_image(image, kernel_size)

                # Define output path
                output_image_path = os.path.join(output_subdir, image_name)

                # Save the processed image
                cv2.imwrite(output_image_path, image)
    for subdir in ['train/mask', 'test/mask']:
        input_subdir = os.path.join(input_folder, subdir)
        output_subdir = os.path.join(output_folder, subdir)

        # Ensure the output directory exists
        os.makedirs(output_subdir, exist_ok=True)

        # Process each image in the subdirectory
        for mask_name in os.listdir(input_subdir):
            mask_path = os.path.join(input_subdir, mask_name)
        mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(mask_path, mask)


input_folder= "/data/nhthach/project/DATA/RETINAL/Binary/binary_0.15_crop_aug_no_equal_no_norm"
output_folder= "/data/nhthach/project/DATA/RETINAL/Binary/binary_0.15_crop_aug_no_equal_no_norm_erosion"

process_images(input_folder, output_folder, True, kernel_size=(5,5))

