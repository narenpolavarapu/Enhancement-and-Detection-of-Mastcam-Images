import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image from your dataset
image = cv2.imread('i8.jpg', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

# Apply Sobel edge detection in both x and y directions
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y direction

# Combine the Sobel images to get the edge magnitude
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Convert to 8-bit for visualization
sobel_combined = np.uint8(np.absolute(sobel_combined))

# Invert to make edges black and the background white
sobel_black_edges = cv2.threshold(sobel_combined, 30, 255, cv2.THRESH_BINARY_INV)[1]

# Display the original and edge-detected images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_black_edges, cmap='gray')
plt.title('Edge Detected (Black Edges)')
plt.axis('off')

plt.show()
