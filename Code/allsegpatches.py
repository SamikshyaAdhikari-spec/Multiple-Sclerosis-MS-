import cv2
import os
import numpy as np

# Define base directories
base_dir = "C:/Users/samik/Documents/GitHub/MS-disease"
annotation_base_dir = "C:/Users/samik/Desktop/Segmentedpatches"

# Define parameters
num_patients = 60
window_width = 146
window_height = 81
stride_x = 36
stride_y = 20

# Function to detect lesions in `Flair-seg/` using HSV white detection
def contains_lesion(patch):
    """Detects lesion using HSV color thresholding in `Flair-seg/`."""
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Define white color range in HSV
    lower_white = np.array([0, 0, 210])  # Low saturation, high value
    upper_white = np.array([180, 40, 255])  # Low saturation, max value

    # Create a mask for white regions
    white_mask = cv2.inRange(patch_hsv, lower_white, upper_white)

    # Count white pixels in the mask
    white_pixel_count = cv2.countNonZero(white_mask)

    # Detect lesion if enough white pixels exist
    return white_pixel_count > 0  # Adjust threshold if needed

# Loop through each patient
for patient_id in range(1, num_patients + 1):
    patient_folder = f"Patient-{patient_id}"
    flair_seg_path = os.path.join(base_dir, patient_folder, "Flair-seg")  # Lesion mask folder

    # Ensure necessary folder exists
    if not os.path.exists(flair_seg_path):
        print(f"Skipping {patient_folder}: Missing `Flair-seg/` folder.")
        continue

    # Get all slice filenames
    slice_filenames = sorted([f for f in os.listdir(flair_seg_path) if f.endswith('.jpg')])

    # Process each slice
    for slice_filename in slice_filenames:
        seg_image_path = os.path.join(flair_seg_path, slice_filename)  # Flair-seg image

        # Ensure the segmentation image exists
        if not os.path.exists(seg_image_path):
            print(f"Skipping {slice_filename} for {patient_folder} (missing file)")
            continue

        # Load segmentation image
        seg_image = cv2.imread(seg_image_path, cv2.IMREAD_COLOR)  # Load in color for HSV processing

        # Get dimensions
        image_height, image_width, _ = seg_image.shape

        # Create annotation directories
        slice_id = slice_filename.split('.')[0]
        output_dir0 = os.path.join(annotation_base_dir, f"pat{patient_id}_{slice_id}_segmented", "0")
        output_dir1 = os.path.join(annotation_base_dir, f"pat{patient_id}_{slice_id}_segmented", "1")
        os.makedirs(output_dir0, exist_ok=True)
        os.makedirs(output_dir1, exist_ok=True)

        # Initialize patch counter
        patch_counter = 0

        # Sliding window approach on `Flair-seg/`
        for y in range(0, image_height - window_height + 1, stride_y):
            for x in range(0, image_width - window_width + 1, stride_x):
                # Extract patch from `Flair-seg/`
                seg_patch = seg_image[y:y + window_height, x:x + window_width]

                # Detect lesion using HSV method on `Flair-seg/`
                if contains_lesion(seg_patch):
                    patch_dir = output_dir1  # Save in 1/ (lesion)
                else:
                    patch_dir = output_dir0  # Save in 0/ (no lesion)

                # Save patch from `Flair-seg/`
                patch_filename = f"patch_{patch_counter:04d}.jpg"
                patch_path = os.path.join(patch_dir, patch_filename)
                cv2.imwrite(patch_path, seg_patch)

                print(f"Saved segmented patch at ({x}, {y}) for {patient_folder}, {slice_id} to {patch_path}")

                patch_counter += 1

        print(f"Processed {slice_filename} for {patient_folder}. Total patches: {patch_counter}")

print("All Flair-seg segmented patches extracted successfully! 🎉")
