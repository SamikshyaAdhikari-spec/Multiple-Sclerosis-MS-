import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Define the base directory containing all patient folders
base_dir = "C:/Users/samik/Documents/GitHub/MS-disease"  # Replace with the base directory

smallest_lesion_area = float('inf')  # Initialize with infinity
smallest_lesion_image = None
smallest_lesion_patient = None

# Loop through all patient folders (Patient-1 to Patient-60)
for patient_id in range(1, 61):  # Adjust range for the total number of patients
    patient_folder = os.path.join(base_dir, f"Patient-{patient_id}")
    flair_seg_folder = os.path.join(patient_folder, "Flair-seg")

    if os.path.exists(flair_seg_folder):  # Check if Flair-seg folder exists
        # Loop through all images in the Flair-seg folder
        for filename in os.listdir(flair_seg_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
                image_path = os.path.join(flair_seg_folder, filename)

                # Load the image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Threshold the image to ensure it's binary
                # This ensures all lesion becomes pure white and background becomes pure black
                _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

                # Label connected components
                # detects distinct lesions in the binary image
                # stats contains properties of each detected region
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

                # Check if any lesions (white regions) exist
                if num_labels > 1:  # More than one label means there's at least one lesion (excluding background)
                    # Find the area of each component (excluding the background)
                    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the first component (background)
                    min_area = min(areas)

                    # Check if this lesion is the smallest across all images
                    if min_area < smallest_lesion_area and min_area > 0:
                        smallest_lesion_area = min_area
                        smallest_lesion_image = image_path
                        smallest_lesion_patient = f"Patient-{patient_id}"

# Output the results

if smallest_lesion_image:
    print(f"Smallest lesion found in: {smallest_lesion_image}")
    print(f"Smallest lesion area: {smallest_lesion_area} pixels")
    print(f"Smallest lesion belongs to: {smallest_lesion_patient}")

    # Load and display the image with the smallest lesion
    smallest_image = cv2.imread(smallest_lesion_image, cv2.IMREAD_GRAYSCALE)
    _, binary_smallest = cv2.threshold(smallest_image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_smallest, connectivity=8)

    # Find the mask for the smallest lesion
    areas = stats[1:, cv2.CC_STAT_AREA]
    smallest_lesion_index = np.argmin(areas) #Gets the index of the smallest lesion
    smallest_lesion_mask = (labels == (smallest_lesion_index + 1)).astype(np.uint8) * 255

    # Display the original image and the smallest lesion mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(smallest_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Smallest Lesion (Area: {smallest_lesion_area} pixels)")
    plt.imshow(smallest_lesion_mask, cmap='gray')
    plt.axis('off')

    plt.show()

else:
    print("No lesions found in any of the images.")