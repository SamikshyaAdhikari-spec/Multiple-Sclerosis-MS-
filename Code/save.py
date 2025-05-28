import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Load the dataset
file_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/1-Flair.nii'
img = nib.load(file_path)

# Get the image data as a numpy array
img_data = img.get_fdata()


# Define the function to visualize and save each slice
def visualize_and_save_slices(data, output_dir="output_slices", delay=1):
    """
    Visualize and save slices of a 3D volume as images.

    Parameters:
        data (numpy array): 3D image data (X, Y, Z).
        output_dir (str): Directory to save the slices.
        delay (int): Delay (in seconds) between displaying slices.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving slices to: {output_dir}")

    slices = data.shape[2]  # Number of slices along the z-axis

    for i in range(slices):
        # Extract the i-th slice
        slice_data = data[:, :, i]

        # Plot the slice
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap='gray')
        plt.title(f'Slice {i + 1}')
        plt.axis('off')

        # Save the slice as a PNG file
        output_path = os.path.join(output_dir, f"slice_{i + 1:03}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved slice {i + 1} to {output_path}")

        # Show the slice
        plt.show()

        # Wait for a specified time
        time.sleep(delay)
        plt.close()  # Close the figure to save memory


# Set the output directory
output_directory = 'C:/Users/samik/Desktop/output_slices'

# Call the function to visualize and save slices
visualize_and_save_slices(img_data, output_dir=output_directory, delay=1)
