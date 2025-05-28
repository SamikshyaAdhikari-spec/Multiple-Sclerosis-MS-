import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Define the base directory containing all patient folders
base_dir = "C:/Users/samik/Documents/GitHub/MS-disease"  # Replace with the base directory

largest_lesion_area = 0
largest_lesion_image = None
largest_lesion_patient = None

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
                _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

                # Label connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

                # Check if any lesions (white regions) exist
                if num_labels > 1:  # More than one label means there's at least one lesion (excluding background)
                    # Find the area of each component (excluding the background)
                    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the first component (background)
                    max_area = max(areas)

                    # Check if this lesion is the largest across all images
                    if max_area > largest_lesion_area:
                        largest_lesion_area = max_area
                        largest_lesion_image = image_path
                        largest_lesion_patient = f"Patient-{patient_id}"

# Output the results

if largest_lesion_image:
    print(f"Largest lesion found in: {largest_lesion_image}")
    print(f"Largest lesion area: {largest_lesion_area} pixels")
    print(f"Largest lesion belongs to: {largest_lesion_patient}")

    # Load and display the image with the largest lesion
    largest_image = cv2.imread(largest_lesion_image, cv2.IMREAD_GRAYSCALE)
    _, binary_largest = cv2.threshold(largest_image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_largest, connectivity=8)

    # Find the mask for the largest lesion
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_lesion_index = np.argmax(areas)
    largest_lesion_mask = (labels == (largest_lesion_index + 1)).astype(np.uint8) * 255

    # Display the original image and the largest lesion mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(largest_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Largest Lesion (Area: {largest_lesion_area} pixels)")
    plt.imshow(largest_lesion_mask, cmap='gray')
    plt.axis('off')

    plt.show()

else:
    print("No lesions found in any of the images.")
 # Bpunding box code
 #image_path largest lesion img
 #label.py
 #width = w *2
 #height = h * 2
 #stride = w/4 h/4

 #70/30 split
 #Train model
 #Test model