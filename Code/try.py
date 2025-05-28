import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the base directory
base_dir = "C:/Users/samik/Documents/GitHub/MS-disease"

# Define the number of patients
num_patients = 60

# Loop through each patient
for patient_id in range(1, num_patients + 1):
    patient_folder = f"Patient-{patient_id}"
    flair_path = os.path.join(base_dir, patient_folder, "Flair")
    flair_seg_path = os.path.join(base_dir, patient_folder, "Flair-seg")
    output_path = os.path.join(base_dir, patient_folder, "Overlayed")

    # Create output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Get all slice filenames from the Flair folder
    slice_filenames = sorted([f for f in os.listdir(flair_path) if f.endswith('.jpg')])

    # Process each slice
    for slice_filename in slice_filenames:
        original_image_path = os.path.join(flair_path, slice_filename)
        segmented_image_path = os.path.join(flair_seg_path, slice_filename)

        # Ensure both original and segmented images exist
        if not os.path.exists(original_image_path) or not os.path.exists(segmented_image_path):
            print(f"Skipping {slice_filename} for {patient_folder} (missing file)")
            continue

        # Read images
        original = cv2.imread(original_image_path)
        segmented = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)

        # Resize the segmented image to match the original image if necessary
        segmented_resized = cv2.resize(segmented, (original.shape[1], original.shape[0]))

        # Apply color map to the segmented image
        segmented_colored = cv2.applyColorMap(segmented_resized, cv2.COLORMAP_JET)

        # Overlay the segmented lesion onto the original image
        alpha = 0.6  # Transparency factor
        overlay = cv2.addWeighted(segmented_colored, alpha, original, 1 - alpha, 0)

        # Save the overlay image
        overlay_output_path = os.path.join(output_path, slice_filename)
        cv2.imwrite(overlay_output_path, overlay)

    print(f"Processed Patient-{patient_id}")

print("All patients processed successfully.")
