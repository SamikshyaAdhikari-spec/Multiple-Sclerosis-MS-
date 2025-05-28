import cv2
import os

# Path to the input image
image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/slice_13new.jpg"  # Replace with the actual image path

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the sliding window size
window_width = 146
window_height = 81

# Define the stride size
stride_x = 36  # Horizontal stride
stride_y = 20  # Vertical stride

# Get the dimensions of the image
image_height, image_width = image.shape

# Directory to save the patches
output_dir = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/patchesnew"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Counter for naming patches
patch_counter = 0

# Iterate through the image using the sliding window
for y in range(0, image_height - window_height + 1, stride_y):
    for x in range(0, image_width - window_width + 1, stride_x):
        # Extract the window
        patch = image[y:y + window_height, x:x + window_width]

        # Save the patch
        patch_filename = f"patch_{patch_counter:04d}.jpg"  # Example: patch_0001.jpg
        patch_path = os.path.join(output_dir, patch_filename)
        cv2.imwrite(patch_path, patch)

        print(f"Saved patch at ({x}, {y}) to {patch_path}")

        # Increment the patch counter
        patch_counter += 1

print(f"Total patches saved: {patch_counter}")
