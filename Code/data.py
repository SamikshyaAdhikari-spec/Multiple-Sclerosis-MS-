import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time

def load_nii_data(file_path):
    """Load .nii file and return its data"""
    img = nib.load(file_path)
    return img.get_fdata()

def normalize_slice(slice_data):
    """Normalize slice intensities to the 0-255 range if possible"""
    min_val = slice_data.min()
    max_val = slice_data.max()
    if max_val - min_val == 0:
        return np.zeros(slice_data.shape, dtype=np.uint8)
    return ((slice_data - min_val) * 255 / (max_val - min_val)).astype(np.uint8)

def register_volume(original_vol, segmented_vol, annotations):
    """Register segmented volume to original volume using annotations"""
    registered_vol = np.zeros_like(segmented_vol)
    
    for slice_idx in range(original_vol.shape[2]):
        original_slice = original_vol[:, :, slice_idx]
        segmented_slice = segmented_vol[:, :, slice_idx]
        
        # Normalize slice intensities
        original_norm = normalize_slice(original_slice)
        segmented_norm = normalize_slice(segmented_slice)
        
        # Register slice using annotations
        if slice_idx in annotations:
            registered_slice = register_slice_with_annotations(original_norm, segmented_norm, annotations[slice_idx])
        else:
            registered_slice = segmented_norm  # No registration, use original segmented slice
        
        registered_vol[:, :, slice_idx] = registered_slice
    
    return registered_vol

def register_slice_with_annotations(original_slice, segmented_slice, annotation):
    """Align segmented slice to original slice based on given annotation coordinates"""
    if len(annotation['original']) < 4 or len(annotation['segmented']) < 4:
        print(f"Skipping slice due to insufficient points.")
        return segmented_slice
    
    # Annotation contains coordinates as (x, y) pairs for alignment
    src_pts = np.float32(annotation['segmented']).reshape(-1, 1, 2)
    dst_pts = np.float32(annotation['original']).reshape(-1, 1, 2)
    
    # Calculate homography matrix from annotated points
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print(f"Homography calculation failed for slice.")
        return segmented_slice
    
    registered = cv2.warpPerspective(segmented_slice, H, 
                                     (original_slice.shape[1], original_slice.shape[0]))
    
    return registered

def get_lesion_coordinates(binary_mask):
    """Find lesion coordinates in a binary mask using contour detection."""
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_coords = []
    for contour in contours:
        for point in contour:
            x, y = point[0]  # Extract x, y coordinates
            lesion_coords.append((x, y))
    return lesion_coords

def save_all_slices(original_vol, registered_vol, output_dir="output_slices"):
    """Save each axial slice with overlay as an image file in the specified directory"""
    os.makedirs(output_dir, exist_ok=True)
    num_slices = original_vol.shape[2]
    
    for slice_idx in range(num_slices):
        orig_slice = original_vol[:, :, slice_idx]
        reg_slice = registered_vol[:, :, slice_idx]
        
        # Normalize and overlay the registered slice on the original slice
        overlay = orig_slice.copy()
        overlay[reg_slice > 0] = np.max(orig_slice)
        
        # Plot the overlayed slice
        plt.figure(figsize=(6, 6))
        plt.imshow(orig_slice, cmap='gray')
        plt.imshow(reg_slice > 0, cmap='hot', alpha=0.3)
        plt.axis('off')
        
        # Save the image
        output_path = os.path.join(output_dir, f'slice_{slice_idx + 1}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Saved slice {slice_idx + 1} as {output_path}")

def main():
    original_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-60/60-Flair.nii'
    segmented_path = 'C:/Users/samik/Documents/GitHub/MS-disease/Patient-60/60-LesionSeg-Flair.nii'
    
    # Example of annotations: a dictionary where each slice index has a list of corresponding points
    annotations = {
        0: {'original': [(30, 40), (80, 90), (50, 60), (70, 80)], 'segmented': [(32, 42), (82, 92), (52, 62), (72, 82)]},
        # Add annotations for each slice with at least four points
    }
    
    print("Loading volumes...")
    original_vol = load_nii_data(original_path)
    segmented_vol = load_nii_data(segmented_path)
    
    print("Performing registration with annotations...")
    registered_vol = register_volume(original_vol, segmented_vol, annotations)
    
    # Extract lesion coordinates from the segmented and registered images
    for slice_idx in range(original_vol.shape[2]):
        segmented_slice = normalize_slice(segmented_vol[:, :, slice_idx])
        registered_slice = normalize_slice(registered_vol[:, :, slice_idx])
        
        # Create binary masks for lesions
        segmented_mask = segmented_slice > 0
        registered_mask = registered_slice > 0
        
        # Get lesion coordinates in both images
        segmented_coords = get_lesion_coordinates(segmented_mask)
        registered_coords = get_lesion_coordinates(registered_mask)
        
        print(f"Slice {slice_idx + 1}:")
        print(f" - Original Lesion Coordinates (Segmented Image): {segmented_coords[:5]}...")  # Display first 5 for brevity
        print(f" - Registered Lesion Coordinates: {registered_coords[:5]}...")
    
    print("Saving all slices with overlay...")
    save_all_slices(original_vol, registered_vol, output_dir="output_slices")

if __name__ == "__main__":
    main()
