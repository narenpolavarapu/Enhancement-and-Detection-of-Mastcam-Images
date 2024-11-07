import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image from your dataset
image = cv2.imread('i4.jpg', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

# Prewitt operator (calculating both x and y directions)
prewitt_x = cv2.filter2D(image, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
prewitt_y = cv2.filter2D(image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))

# Combine the Prewitt images to get the edge magnitude
prewitt_combined = cv2.magnitude(prewitt_x.astype(np.float32), prewitt_y.astype(np.float32))
prewitt_combined = np.uint8(prewitt_combined)  # Convert to 8-bit for display

# Display the original and Prewitt edge-detected image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(prewitt_combined, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
