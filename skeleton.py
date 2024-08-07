
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Augted)vein/Train/VeinMask/1282_84_5.png'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image
# Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Use adaptive thresholding to segment the veins
ret, thresh = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)


# Skeletonize the image using morphological operations
def skeletonize(img):
    size = np.size(img)
    skeleton = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    finished = False

    while not finished:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            finished = True

    return skeleton


skeleton = skeletonize(thresh)

# Detect edges using Canny edge detection
edges = cv2.Canny(thresh, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store the vein widths
vein_widths = []

# Measure the width of each vein segment based on the skeleton
for contour in contours:
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    skel_contour = cv2.bitwise_and(skeleton, mask)

    # Find the width of the skeleton within the contour
    points = np.column_stack(np.where(skel_contour > 0))
    if points.size > 0:
        x_coords = points[:, 1]
        vein_width = np.max(x_coords) - np.min(x_coords)
        vein_widths.append(vein_width)

# Convert the list of widths to a NumPy array (tensor)
vein_widths_tensor = np.array(vein_widths)

# Display the original image, the edges, and the skeleton for verification
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Detected Edges')
plt.imshow(edges, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Skeleton of Veins')
plt.imshow(skeleton, cmap='gray')
plt.show()

# Print the vein widths tensor
print("Vein Widths Tensor:", vein_widths_tensor)