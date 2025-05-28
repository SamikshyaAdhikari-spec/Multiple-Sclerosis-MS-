import os
import shutil

# Define the source directory where the patient slice folders are located
source_directory = "C:/Users/samik/Documents/GitHub/MS-disease/Originalpatches"  # Change this to the actual directory

# Define the destination directory for classification
destination_directory = os.path.join(source_directory, "classification")

# Create classification folder with subfolders for '0' and '1'
os.makedirs(os.path.join(destination_directory, "0"), exist_ok=True)
os.makedirs(os.path.join(destination_directory, "1"), exist_ok=True)

# Iterate through all patient slice folders
for patient_slice in os.listdir(source_directory):
    patient_slice_path = os.path.join(source_directory, patient_slice)

    # Check if it's a directory and follows the naming convention
    if os.path.isdir(patient_slice_path) and "original" in patient_slice:
        for lesion_type in ["0", "1"]:  # '0' for no lesion, '1' for lesion
            lesion_folder = os.path.join(patient_slice_path, lesion_type)

            if os.path.exists(lesion_folder):
                for image in os.listdir(lesion_folder):
                    image_path = os.path.join(lesion_folder, image)

                    if os.path.isfile(image_path):
                        # Rename the image using patient slice folder name
                        new_image_name = f"{patient_slice}_{image}"
                        destination_path = os.path.join(destination_directory, lesion_type, new_image_name)

                        # Copy the image to the corresponding classification folder
                        shutil.copy(image_path, destination_path)

print("Images copied and renamed successfully!")
