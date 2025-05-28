import cv2
import numpy as np
import pandas as pd
import os

# Define root directory containing all patient folders
root_directory = "C:/Users/samik/Documents/GitHub/MS-disease/"  # Adjust this path accordingly

# Output paths
output_folder = os.path.join(root_directory, "processed_lesions1")  # Folder to save processed images
csv_path = os.path.join(root_directory, "lesion_areas1.csv")  # CSV file for results

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Store all lesion data
all_lesion_data = []

# Loop through all patient folders (Patient-1 to Patient-60)
for patient_folder in sorted(os.listdir(root_directory)):
    patient_path = os.path.join(root_directory, patient_folder)

    # Check if the path is a directory and follows the "Patient-X" format
    if os.path.isdir(patient_path) and patient_folder.startswith("Patient-"):

        # Define the "Flair-seg" directory inside each patient folder
        flair_seg_path = os.path.join(patient_path, "Flair-seg")

        # Check if the "Flair-seg" folder exists
        if os.path.exists(flair_seg_path):

            # Process each lesion image inside the "Flair-seg" folder
            for filename in sorted(os.listdir(flair_seg_path)):
                if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only images
                    image_path = os.path.join(flair_seg_path, filename)
                    output_image_path = os.path.join(output_folder, f"{patient_folder}_{filename}")  # Save processed image

                    # Extract slice name from filename (e.g., "slice_17.jpg" -> "slice_17")
                    slice_name = os.path.splitext(filename)[0]

                    # Load image in grayscale
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    # Convert to binary image (Thresholding), converting image into black and white
                    #Pixel gretaer than 127 is set to 255(white)
                    #Pixel values < 127 → Set to 0(Black, background).
                    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

                    # Find connected components
                    #connectedComponentsWithStats is used to identify separate lesions
                    #seperates white lesions in binary image
                    #Labels each connected lesions with a unique number (1 to num_labels - 1).
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

                    # Convert grayscale to BGR (so we can draw colored rectangles)
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                    # Process detected lesions (excluding background)
                    for i in range(1, num_labels):
                        x, y, w, h, area = stats[i]  # Bounding box coordinates and area
                        lesion_name = f"{patient_folder}_{slice_name}_Lesion_{i}"  # Unique lesion name
                        bounding_box = f"({x}, {y}, {w}, {h})"  # Store coordinates as a string

                        # Store lesion details
                        all_lesion_data.append([patient_folder, slice_name, lesion_name, i, area, w, h, bounding_box])

                        # Draw bounding box
                        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green box

                    # Save processed image
                    cv2.imwrite(output_image_path, image_bgr)
                    print(f"Processed image saved: {output_image_path}")

# Create a DataFrame for all lesion data
df_lesions = pd.DataFrame(all_lesion_data, columns=[
    "Patient", "Slice Name", "Lesion Name", "Lesion Index",
    "Area (pixels)", "Width", "Height", "Bounding Box (x, y, w, h)"
])

# Save lesion data to CSV
df_lesions.to_csv(csv_path, index=False)
print(f"Lesion data saved to: {csv_path}")

# Print lesion data summary
print(df_lesions)
