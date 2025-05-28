import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original and segmented images
original_image_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-42/Flair/slice_12.jpg'  # Original slice
segmented_image_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-42/Flair-seg/slice_12.jpg'  # Segmented lesion

original = cv2.imread(original_image_path)
segmented = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)

# Resize the segmented image to match the original image if they have different sizes
segmented_resized = cv2.resize(segmented, (original.shape[1], original.shape[0]))

# Create a color map for the segmented image
# converts grayscale seg mask into color coded heatmap
# highlights them into color
segmented_colored = cv2.applyColorMap(segmented_resized, cv2.COLORMAP_JET)

# Overlay the segmented lesion onto the original image
# 60% contribution of red lesion and 40% concentration of original image
alpha = 0.6  # Transparency factor for overlay
# Blends the image so segmented images are on top of the original MRI image
overlay = cv2.addWeighted(segmented_colored, alpha, original, 1 - alpha, 0)

# Display the original image
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# Display the segmented mask before overlay
plt.figure(figsize=(8, 8))
plt.imshow(segmented_resized, cmap='gray')
plt.title("Segmented Mask")
plt.axis('off')
plt.show()

# Display the overlaid image
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Overlay of Original and Segmented Images")
plt.axis('off')
plt.show()

# Save the overlaid image
output_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-42/slice_12new.jpg'
cv2.imwrite(output_path, overlay)
print(f"Overlaid image saved at: {output_path}")
