import cv2
import os
import pandas as pd
import ast

# Define root directories
processed_images_folder = "C:/Users/samik/Documents/GitHub/MS-disease/processed_lesions1"  # Adjust as needed
output_lesion_folder = "C:/Users/samik/Documents/GitHub/MS-disease/extracted_lesions"
largest_lesion_folder = "C:/Users/samik/Documents/GitHub/MS-disease/largest"
smallest_lesion_folder = "C:/Users/samik/Documents/GitHub/MS-disease/smallest"

# Create necessary directories
os.makedirs(output_lesion_folder, exist_ok=True)
os.makedirs(largest_lesion_folder, exist_ok=True)
os.makedirs(smallest_lesion_folder, exist_ok=True)

# Load lesion CSV
csv_path = "C:/Users/samik/Documents/GitHub/MS-disease/lesion_areas1.csv"
df_lesions = pd.read_csv(csv_path)

# Get all available files in processed_lesions
available_files = set(os.listdir(processed_images_folder))

# Possible extensions
extensions = [".png", ".jpg"]

# Process each lesion
for _, row in df_lesions.iterrows():
    patient = row['Patient']
    slice_name = row['Slice Name']
    lesion_name = row['Lesion Name']
    bounding_box = ast.literal_eval(row['Bounding Box (x, y, w, h)'])  # Convert string to tuple making it easier to use
    x, y, w, h = bounding_box
    area = row['Area (pixels)']

    # Construct possible filenames
    found = False
    for ext in extensions:
        #checks whether patientID and slice name exists in processed_images_folder
        image_filename = f"{patient}_{slice_name}{ext}"
        image_path = os.path.join(processed_images_folder, image_filename)

        if image_filename in available_files:
            # Load processed image
            image = cv2.imread(image_path)

            if image is not None:
                # Crop lesion
                #Crops the lesion using the bounding box (x, y, w, h).

                lesion_crop = image[y:y + h, x:x + w]

                # Define output path for lesion image
                lesion_output_path = os.path.join(output_lesion_folder, f"{lesion_name}{ext}")

                # Save cropped lesion image
                cv2.imwrite(lesion_output_path, lesion_crop)
                print(f"Saved lesion: {lesion_output_path}")
                found = True
                break

    if not found:
        print(f"Image not found for: {patient}_{slice_name}")

# Sort lesions by area and get top/bottom 10 using csv file
largest_lesions = df_lesions.nlargest(10, 'Area (pixels)')
smallest_lesions = df_lesions.nsmallest(10, 'Area (pixels)')

# Copy images to respective folders
for _, row in largest_lesions.iterrows():
    lesion_name = row['Lesion Name']
    for ext in extensions:
        lesion_path = os.path.join(output_lesion_folder, f"{lesion_name}{ext}")
        if os.path.exists(lesion_path):
            cv2.imwrite(os.path.join(largest_lesion_folder, f"{lesion_name}{ext}"), cv2.imread(lesion_path))
            break

for _, row in smallest_lesions.iterrows():
    lesion_name = row['Lesion Name']
    for ext in extensions:
        lesion_path = os.path.join(output_lesion_folder, f"{lesion_name}{ext}")
        if os.path.exists(lesion_path):
            cv2.imwrite(os.path.join(smallest_lesion_folder, f"{lesion_name}{ext}"), cv2.imread(lesion_path))
            break

print("Lesion extraction complete. Largest and smallest lesions separated.")
