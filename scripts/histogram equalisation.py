import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'i2.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Split the image into its RGB components
b, g, r = cv2.split(image)

# Equalize each channel individually
r_eq = cv2.equalizeHist(r)
g_eq = cv2.equalizeHist(g)
b_eq = cv2.equalizeHist(b)

# Merge the equalized channels back together
equalized_image = cv2.merge((b_eq, g_eq, r_eq))

# Plotting the original and equalized images with their histograms
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
plt.title('Original Image')
plt.axis('off')

# Equalized Image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Image')
plt.axis('off')

# Histogram for Original Image
plt.subplot(2, 2, 3)
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
plt.title('Histogram for Original Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid()

# Histogram for Equalized Image
plt.subplot(2, 2, 4)
for i, color in enumerate(colors):
    histogram = cv2.calcHist([equalized_image], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
plt.title('Histogram for Equalized Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid()

plt.tight_layout()
plt.show()
