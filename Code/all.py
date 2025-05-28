import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math


# Define the base directory containing all patient folders
base_dir = "C:/Users/samik/Documents/GitHub/MS-disease"  # Replace with the base directory

largest_lesion_area = 0
largest_lesion_image = None
largest_lesion_patient = None

# Loop through all patient folders (Patient-1 to Patient-60)
for patient_id in range(1, 61):  # Adjust range for the total number of patients
    patient_folder = os.path.join(base_dir, f"Patient-{patient_id}")
    flair_seg_folder = os.path.join(patient_folder, "Flair-seg") #Checks Flair-seg for finding largest lesion

    if os.path.exists(flair_seg_folder):  # Check if Flair-seg folder exists
        # Loop through all images in the Flair-seg folder
        for filename in os.listdir(flair_seg_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
                image_path = os.path.join(flair_seg_folder, filename)
#....Patient 1/Flair-seg/Slice_13.png
                # Load the image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Threshold the image to ensure it's binary
                #Converts grey scale image to black and white
                # any pixel intensity greater than 127 is set to max 255 i.e. white and others to 0 i.e black
                _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

                # Label connected components
                #Identify connected white pixels
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

                # Check if any lesions (white regions) exist
                if num_labels > 1:  # More than one label means there's at least one lesion (excluding background)
                    # Find the area of each component (excluding the background)
                    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the first component (background)
                    max_area = max(areas) # Finds max area in each

                    # Check if this lesion is the largest across all images
                    if max_area > largest_lesion_area:
                        largest_lesion_area = max_area
                        largest_lesion_image = image_path
                        largest_lesion_patient = f"Patient-{patient_id}"

# If a largest lesion is found, calculate the bounding box dimensions
if largest_lesion_image:
    print(f"Largest lesion found in: {largest_lesion_image}")
    print(f"Largest lesion area: {largest_lesion_area} pixels")
    print(f"Largest lesion belongs to: {largest_lesion_patient}")

    # Process the image containing the largest lesion to calculate bounding box
    image = cv2.imread(largest_lesion_image, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to ensure it's binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) #This code creates binary mask

    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Find the largest lesion
    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the first component (background)
    max_index = np.argmax(areas)  # Index of the largest lesion

    # Get the bounding box dimensions
    x = stats[max_index + 1, cv2.CC_STAT_LEFT]
    y = stats[max_index + 1, cv2.CC_STAT_TOP]
    w = stats[max_index + 1, cv2.CC_STAT_WIDTH]
    h = stats[max_index + 1, cv2.CC_STAT_HEIGHT]

    print(f"Bounding box width: {w} pixels")
    print(f"Bounding box height: {h} pixels")

    # Dynamic sliding window parameters
    window_width = w  # Use the width of the bounding box
    window_height = h  # Use the height of the bounding box
    # Calculate strides as one-fourth of the window dimensions, rounded down
    stride_x = math.floor(window_width / 8)
    stride_y = math.floor(window_height / 8)

    print(f"Sliding window dimensions: {window_width}x{window_height}")
    print(f"Stride dimensions: {stride_x}x{stride_y}")

    # Path to the input image
    image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-42/Flair-seg/slice_12.jpg"
    original_image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-42/Flair/slice_12.jpg"

    # Load the images
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)

    # Resize Flair-seg image to match the original Flair image dimensions
    image = cv2.resize(image, (original_image.shape[1], original_image.shape[0]))

    # Get the dimensions of the image
    image_height, image_width, _ = image.shape

    # Directories to save the patches
    output_dir0 = "C:/Users/samik/Documents/GitHub/MS-disease/whitelabel/0"
    output_dir1 = "C:/Users/samik/Documents/GitHub/MS-disease/whitelabel/1"
    os.makedirs(output_dir0, exist_ok=True)
    os.makedirs(output_dir1, exist_ok=True)

    # Counter for naming patches
    patch_counter = 0

    # Function to check if a patch contains a lesion which in this case is white regions
    def contains_lesion(patch): #takes segmented image patches
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV) #Converted to HSV because easier to detect white in the image
        lower_white = np.array([0, 0, 200]) # For capturing white like pixel
        upper_white = np.array([180, 50, 255]) # For capturing white-like pixel
        white_mask = cv2.inRange(patch_hsv, lower_white, upper_white) #Compare patches in patch_hsv with white like
        white_pixel_count = cv2.countNonZero(white_mask) #Counts no of white pixels
        return white_pixel_count > 0 #lesions(white color) > 0 returns True

    # Sliding window operation
    #image_width - window_width + 1 ensures sliding window doesnot go outside the image boundary
    for y in range(0, image_height - window_height + 1, stride_y):
        for x in range(0, image_width - window_width + 1, stride_x):
            patch = image[y:y + window_height, x:x + window_width] #Segmented image are extracted in patch
            original_patch = original_image[y:y + window_height, x:x + window_width] #Original image are extracted in patch

            if contains_lesion(patch):
                patch_dir = output_dir1 #if patch has lesion then it is saved in output_dir 1
            else:
                patch_dir = output_dir0 #Else save in output_dir 2

            patch_filename = f"patch_{patch_counter:04d}.jpg"
            patch_path = os.path.join(patch_dir, patch_filename) #Sets patches as patch_0001...
            cv2.imwrite(patch_path, original_patch) #Original image patch is used here

            print(f"Saved patch at ({x}, {y}) to {patch_path}")

            patch_counter += 1 #Ensures each patch gets a unique name after saving each patch

    print(f"Total patches saved: {patch_counter}")

else:
    print("No lesions found in any of the images.")
