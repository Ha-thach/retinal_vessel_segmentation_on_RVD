
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/data/nhthach/project/RETINAL/DATA/subset_train_test/train/image/0001_54.png'
image = cv2.imread(image_path)
(b, g, r)=cv2.split(image)
blank = np.zeros_like(b)
    #print("shape of blank", blank.shape)
#blue = cv2.merge([b, blank, blank])

    #red = cv2.merge([blank, blank, r])

green = cv2.merge([g, g, g])
green1= cv2.cvtColor(green,cv2.COLOR_BGR2GRAY)

    #concate_image=np.concatenate([image, blue, green, red], axis=1)

    #  create a one channel image
    #cv2.imwrite(output_path, concate_image)

#cv2.imwrite(output_path, green)

# Convert the image to RGB (OpenCV loads images in BGR format)
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the image
plt.figure()
plt.imshow(green1,cmap='gray')
plt.title('green')
plt.axis('off')  # Turn off axis


# Calculate the histogram
histogram = cv2.calcHist(image, [0], None, [256], [0, 256])

# Plot the histogram

plt.figure()
plt.plot(histogram)
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

ret,thresh1=cv2.threshold(image,18,255,cv2.THRESH_BINARY)
plt.subplot(1,2,1)
plt.imshow(thresh1,cmap='gray')
plt.title('BinaryThresholdandotsu')
plt.axis('off')

cv2.waitKey()
cv2.destroyWindow()
print(green1.shape)

