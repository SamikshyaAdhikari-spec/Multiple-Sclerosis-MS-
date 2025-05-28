import cv2
import numpy as np

# Path to the input image
image_path = "C:/Users/samik/Documents/GitHub/MS-disease/Patient-1/slice_13new.jpg"  # Replace with the actual image path

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the sliding window size
window_width = 146
window_height = 81

# Define the stride size
stride_x = 36  # Horizontal stride
stride_y = 20 # Vertical stride

# Get the dimensions of the image
image_height, image_width = image.shape

# Iterate through the image using the sliding window
for y in range(0, image_height - window_height + 1, stride_y):
    for x in range(0, image_width - window_width + 1, stride_x):
        # Extract the window
        window = image[y:y + window_height, x:x + window_width]

        # Process the window (e.g., display or save it)
        print(f"Window at ({x}, {y}) with size {window_width}x{window_height}")
        # Example: Display the window using OpenCV
        cv2.imshow("Sliding Window", window)
        cv2.waitKey(100)  # Display each window for 100ms (adjust as needed)

# Close all OpenCV windows
cv2.destroyAllWindows()
