import os

def rename_images_with_prefix(directory, prefix):
    """
    Rename all image files in the directory by adding a prefix.
    
    Parameters:
        directory (str): Path to the directory containing the images.
        prefix (str): Prefix to add to the filenames.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if it's a file (skip directories)
        if os.path.isfile(os.path.join(directory, filename)):
            # Create new filename by adding prefix
            new_filename = prefix + filename
            # Rename the file
            os.rename(
                os.path.join(directory, filename), 
                os.path.join(directory, new_filename)
            )
            print(f"Renamed: {filename} -> {new_filename}")

# Example usage
rename_images_with_prefix("C:/Users/samik/Documents/GitHub/MS-disease/Patient-60/output_slices", "pat60_")
