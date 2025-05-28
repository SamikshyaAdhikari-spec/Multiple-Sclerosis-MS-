import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Directory containing the images
image_dir = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-38/Flair-seg"  # Replace with the directory containing your images
largest_lesion_area = 0
largest_lesion_image = None

# Loop through all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
        image_path = os.path.join(image_dir, filename)

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

# Output the results
if largest_lesion_image:
    print(f"Largest lesion found in: {largest_lesion_image}")
    print(f"Largest lesion area: {largest_lesion_area} pixels")

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
