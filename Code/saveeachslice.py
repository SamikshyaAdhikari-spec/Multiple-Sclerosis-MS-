import nibabel as nib
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-60/60-LesionSeg-Flair.nii'
output_folder = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-60/Flair-seg'  # Folder to save the slices

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

img = nib.load(file_path)

# Get the image data as a numpy array
img_data = img.get_fdata()

# Define the function to save each slice as a .jpg file
def save_slices_as_images(data, output_dir):
    slices = data.shape[2]  # Number of slices along the z-axis

    for i in range(slices):
        plt.figure(figsize=(6, 6))
        plt.imshow(data[:, :, i], cmap='gray')
        plt.axis('off')

        # Save the slice as a .jpg file
        output_path = os.path.join(output_dir, f'slice_{i + 1}.jpg')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory

# Call the function to save each slice
save_slices_as_images(img_data, output_folder)

print(f"All slices have been saved in the folder: {output_folder}")