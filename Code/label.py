import cv2
import os
import numpy as np


# Path to the input image
image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/slice_13new.jpg"  # Replace with the actual image path
original_image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/Flair/slice_13.jpg"  # Replace with the actual image path

# Load the image in color
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)


# Define the sliding window size
window_width = 146
window_height = 81

# Define the stride size
stride_x = 36  # Horizontal stride
stride_y = 20  # Vertical stride

# Get the dimensions of the image
image_height, image_width, _ = image.shape

# Directory to save the patches
output_dir0 = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/label/0"
output_dir1 = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/label/1"
os.makedirs(output_dir0, exist_ok=True)  # Create the directory if it doesn't exist
os.makedirs(output_dir1, exist_ok=True)  # Create the directory if it doesn't exist

# Counter for naming patches
patch_counter = 0

# Function to check if a patch contains a lesion
def contains_lesion(patch):
    # Convert the patch to HSV to better detect red
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions
    mask1 = cv2.inRange(patch_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(patch_hsv, lower_red2, upper_red2)

    # Combine the masks
    red_mask = mask1 + mask2

    # Check if there are enough red pixels to consider it a lesion
    red_pixel_count = cv2.countNonZero(red_mask)
    return red_pixel_count > 0  # Adjust this threshold as needed

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
