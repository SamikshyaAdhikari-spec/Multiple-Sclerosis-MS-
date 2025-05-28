import cv2
import os
import numpy as np

# Define base directories
base_dir = "C:/Users/samik/Documents/GitHub/MS-disease"
annotation_base_dir = "C:/Users/samik/Desktop/annotation"

# Define parameters
num_patients = 60
window_width = 146
window_height = 81
stride_x = 36
stride_y = 20


# Function to detect lesions using Flair-seg (grayscale) and Overlayed (HSV for red lesions)
def contains_lesion(seg_patch, overlay_patch):
    """Detects lesion using both Flair-seg (grayscale) and Overlayed (HSV for red regions)."""

    # **1️⃣ Check Lesions in Flair-seg (Grayscale)**
    seg_white_pixel_count = cv2.countNonZero(seg_patch)  # Count non-black pixels

    # **2️⃣ Check Lesions in Overlayed (HSV for Red)**
    patch_hsv = cv2.cvtColor(overlay_patch, cv2.COLOR_BGR2HSV)

    # Define RED color range in HSV
    lower_red1 = np.array([0, 120, 70])  # Lower boundary for red (hue: 0-10)
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])  # Upper boundary for red (hue: 170-180)
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions
    red_mask1 = cv2.inRange(patch_hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(patch_hsv, lower_red2, upper_red2)

    # Combine both red masks
    red_mask = red_mask1 + red_mask2

    # Count red pixels
    red_pixel_count = cv2.countNonZero(red_mask)

    # **3️⃣ Lesion Detected If Either Flair-seg or Red Regions in Overlayed Image**
    return seg_white_pixel_count > 0 or red_pixel_count > 0


# Loop through each patient
for patient_id in range(1, num_patients + 1):
    patient_folder = f"Patient-{patient_id}"
    flair_seg_path = os.path.join(base_dir, patient_folder, "Flair-seg")  # Lesion mask
    overlay_path = os.path.join(base_dir, patient_folder, "Overlayed")  # Overlayed images

    # Ensure necessary folders exist
    if not os.path.exists(flair_seg_path) or not os.path.exists(overlay_path):
        print(f"Skipping {patient_folder}: Missing required folders.")
        continue

    # Get all slice filenames
    slice_filenames = sorted([f for f in os.listdir(flair_seg_path) if f.endswith('.jpg')])

    # Process each slice
    for slice_filename in slice_filenames:
        seg_image_path = os.path.join(flair_seg_path, slice_filename)  # Flair-seg image
        overlay_image_path = os.path.join(overlay_path, slice_filename)  # Corresponding Overlayed image

        # Ensure both images exist
        if not os.path.exists(seg_image_path) or not os.path.exists(overlay_image_path):
            print(f"Skipping {slice_filename} for {patient_folder} (missing files)")
            continue

        # Load images
        seg_image = cv2.imread(seg_image_path, cv2.IMREAD_GRAYSCALE)  # Flair-seg (lesion mask)
        overlayed_image = cv2.imread(overlay_image_path, cv2.IMREAD_COLOR)  # Overlayed image for patches

        # Resize segmentation image to match overlay image
        seg_image = cv2.resize(seg_image, (overlayed_image.shape[1], overlayed_image.shape[0]))

        # Get dimensions
        image_height, image_width = seg_image.shape

        # Create annotation directories
        slice_id = slice_filename.split('.')[0]
        output_dir0 = os.path.join(annotation_base_dir, f"pat{patient_id}_{slice_id}_annoted", "0")
        output_dir1 = os.path.join(annotation_base_dir, f"pat{patient_id}_{slice_id}_annoted", "1")
        os.makedirs(output_dir0, exist_ok=True)
        os.makedirs(output_dir1, exist_ok=True)

        # Initialize patch counter
        patch_counter = 0

        # Sliding window approach on `Flair-seg/`
        for y in range(0, image_height - window_height + 1, stride_y):
            for x in range(0, image_width - window_width + 1, stride_x):
                # Extract patches
                seg_patch = seg_image[y:y + window_height, x:x + window_width]  # Patch from Flair-seg
                overlay_patch = overlayed_image[y:y + window_height,
                                x:x + window_width]  # Corresponding Overlayed patch

                # Detect lesion using both methods
                if contains_lesion(seg_patch, overlay_patch):
                    patch_dir = output_dir1  # Save in 1/ (lesion)
                else:
                    patch_dir = output_dir0  # Save in 0/ (no lesion)

                # Save patch from Overlayed image
                patch_filename = f"patch_{patch_counter:04d}.jpg"
                patch_path = os.path.join(patch_dir, patch_filename)
                cv2.imwrite(patch_path, overlay_patch)

                print(f"Saved patch at ({x}, {y}) for {patient_folder}, {slice_id} to {patch_path}")

                patch_counter += 1

        print(f"Processed {slice_filename} for {patient_folder}. Total patches: {patch_counter}")

print("All patients processed successfully! 🎉")
