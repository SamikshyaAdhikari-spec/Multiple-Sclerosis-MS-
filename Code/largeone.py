import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-48/Flair-seg/slice_17.jpg"  # Replace with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to ensure it's binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Label connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# Check if any lesions (white regions) exist
if num_labels > 1:  # More than one label means there's at least one lesion (excluding background)
    # Find the area of each component (excluding the background)
    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the first component (background)
    largest_lesion_index = np.argmax(areas)  # Index of the largest lesion
    largest_lesion_area = areas[largest_lesion_index]

    # Highlight the largest lesion in the image
    largest_lesion_mask = (labels == (largest_lesion_index + 1)).astype(np.uint8) * 255

    # Show the binary image and the largest lesion mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Segmented Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Largest Lesion (Area: {largest_lesion_area} pixels)")
    plt.imshow(largest_lesion_mask, cmap='gray')
    plt.axis('off')

    plt.show()

else:
    print("No lesions found in the image.")
