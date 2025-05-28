import cv2
import numpy as np
from matplotlib import pyplot as plt

# Path to the image containing the largest lesion
image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-3/Flair-seg/slice_10.jpg"  # Replace with the actual image path

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to ensure it's binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Label connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# Find the largest lesion
areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the first component (background)
min_index = np.argmin(areas)  # Index of the largest lesion

# Get the bounding box dimensions
x = stats[min_index + 1, cv2.CC_STAT_LEFT]
y = stats[min_index + 1, cv2.CC_STAT_TOP]
w = stats[min_index + 1, cv2.CC_STAT_WIDTH]
h = stats[min_index + 1, cv2.CC_STAT_HEIGHT]

# Output the width and height of the bounding box
print(f"Bounding box width: {w} pixels")
print(f"Bounding box height: {h} pixels")

# Load the original image in color to draw the bounding box
original_image = cv2.imread(image_path)

# Draw the bounding box on the image
image_with_bbox = original_image.copy()
cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green bounding box

# Save the image with the bounding box
output_path = "C:/Users/samik/Documents/GitHub/MS-disease/smallest_lesion_with_bbox.jpg" # Replace with your desired output path
cv2.imwrite(output_path, image_with_bbox)
print(f"Image with bounding box saved to: {output_path}")

# Display the image with bounding box
plt.figure(figsize=(10, 5))
plt.title("Image with Bounding Box")
plt.imshow(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
