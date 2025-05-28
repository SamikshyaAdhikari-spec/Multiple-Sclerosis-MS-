import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the dataset
file_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-2/2-Flair.nii'
img = nib.load(file_path)

# Get the image data as a numpy array
img_data = img.get_fdata()

# Define the function to visualize each slice one by one in a loop
def visualize_slices_one_by_one(data, delay=1):
    slices = data.shape[2]  # Number of slices along the z-axis
    
    for i in range(slices):
        plt.figure(figsize=(6, 6))
        plt.imshow(data[:, :, i], cmap='gray')
        plt.title(f'Slice {i + 1}')
        plt.axis('off')
        plt.show()
        
        # Wait for a specified time before showing the next slice
        time.sleep(delay)
        plt.close()  # Close the previous slice figure to avoid overlap

# Call the function to visualize each slice one by one with a delay of 1 second
visualize_slices_one_by_one(img_data, delay=1)
