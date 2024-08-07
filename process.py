import cv2
import matplotlib.pyplot as plt
import numpy as np

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
image_path = '/data/nhthach/project/RETINAL/DATA/train_test_(no_aug)arteryvein/train/IMAGE/0002_0.png'
show_image_from_path(image_path, "original")
image=cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, remove_noise_background = cv2.threshold(gray_image, 15, 0, cv2.THRESH_TOZERO)
# Define a kernel for erosion
#kernel = np.ones((5, 5), np.uint8)

image_cvt=cv2.cvtColor(remove_noise_background, cv2.COLOR_GRAY2BGR)
# Apply binary erosion
#eroded_image = cv2.erode(thresholded_image, kernel, iterations=1)
#plot_comparison(image, "original", remove_noise_background, "remove noise background", image_cvt, "convered")
b, g, r = cv2.split(image_cvt)
channels = {'Blue': b, 'Green': g, 'Red': r}
plot_comparison(b, "Blue", g, "Green", r, "Red")
clahe = cv2.createCLAHE(clipLimit=5)
clahe_img = clahe.apply(g) + 30
clahe1 = cv2.createCLAHE(clipLimit=3)
clahe_img1 = clahe.apply(g) + 30
clahe2 = cv2.createCLAHE(clipLimit=10)
clahe_img2 = clahe.apply(g) + 30
plot_comparison(clahe_img,"30", clahe_img1,"20", clahe_img2,"10")

# for key in channels:
#     #print(channels[key])
#     #print(key)
#     plt.figure(figsize=(10, 5))
#     plt.title(f'{key} Color Histogram')
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')
#     plt.hist(channels[key].ravel(), bins=256)
#     plt.show()