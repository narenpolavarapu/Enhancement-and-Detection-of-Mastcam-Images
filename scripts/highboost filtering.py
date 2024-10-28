import cv2
import numpy as np
import os

# Path to the folder containing images
input_folder = "C:\\Users\\naren\\Downloads\\archive (3)\\Mars Surface and Curiosity Image\\images"  # Replace with the path to your folder
output_folder = "C:\\Users\\naren\\Downloads\\archive (3)\\Mars Surface and Curiosity Image\\sharpened images"  # Replace with the path where you want to save the processed images

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Highboost factor
A = 2  # Highboost factor (A > 1)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.endswith('.JPG') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Read the image (in color by default)
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Apply Gaussian blur to the color image
        blurred = cv2.GaussianBlur(image, (9, 9), 10.0)

        # Highboost filtering: highboost = (A-1) * original + unsharp_mask
        highboost = cv2.addWeighted(image, A, blurred, -(A-1), 0)

        # Save the highboost filtered image in the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, highboost)

        print(f'Processed and saved: {filename}')

print("Processing complete!")
