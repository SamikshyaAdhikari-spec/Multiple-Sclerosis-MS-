import cv2
import os
import numpy as np


# Path to the input image
image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-51/Flair-seg/slice_16.jpg"  # Replace with the actual image path
original_image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-51/Flair/slice_16.jpg"  # Replace with the actual image path

# Load the image in color
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)

# Make the size of the Flair-seg image equal to the size of the original Flair image
image = cv2.resize(image, (original_image.shape[1], original_image.shape[0]))

# Define the sliding window size
window_width = 146
window_height = 81

# Define the stride size
stride_x = 36  # Horizontal stride
stride_y = 20  # Vertical stride

# Get the dimensions of the image
image_height, image_width, _ = image.shape

# Directory to save the patches
output_dir0 = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-51/whitelabel/0"
output_dir1 = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-51/whitelabel/1"
os.makedirs(output_dir0, exist_ok=True)  # Create the directory if it doesn't exist
os.makedirs(output_dir1, exist_ok=True)  # Create the directory if it doesn't exist

# Counter for naming patches
patch_counter = 0

# Function to check if a patch contains a lesion
def contains_lesion(patch):
    # Convert the patch to HSV to better detect white
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Define white color range in HSV
    lower_white = np.array([0, 0, 200])  # Low saturation, high value
    upper_white = np.array([180, 50, 255])  # Low saturation, max value

    # Create a mask for white regions
    white_mask = cv2.inRange(patch_hsv, lower_white, upper_white)

    # Check if there are enough white pixels to consider it a lesion
    white_pixel_count = cv2.countNonZero(white_mask)
    return white_pixel_count > 0  # Adjust this threshold as needed


# Iterate through the image using the sliding window
for y in range(0, image_height - window_height + 1, stride_y):
    for x in range(0, image_width - window_width + 1, stride_x):
        # Extract the window
        patch = image[y:y + window_height, x:x + window_width]
        original_patch = original_image[y:y + window_height, x:x + window_width]

        # Check if the patch contains a lesion
        if contains_lesion(patch):
            patch_dir = output_dir1
        else:
            patch_dir = output_dir0

        # Save the patch
        patch_filename = f"patch_{patch_counter:04d}.jpg"  # Example: patch_0001.jpg
        patch_path = os.path.join(patch_dir, patch_filename)
        cv2.imwrite(patch_path, original_patch)

        print(f"Saved patch at ({x}, {y}) to {patch_path}")

        # Increment the patch counter
        patch_counter += 1

print(f"Total patches saved: {patch_counter}")
