import os
import numpy as np
from PIL import Image

def check_for_zero_nan_values(image_path):
    # Open BMP image
    bmp_image = Image.open(image_path)

    # Convert BMP image to NumPy array
    bmp_array = np.array(bmp_image)

    # Check for 0 or NaN values
    has_zero_values = np.any(bmp_array == 0)
    has_nan_values = np.isnan(bmp_array).any()

    return has_zero_values, has_nan_values

def find_images_with_zero_nan(folder_path):
    # Get a list of all BMP files in the folder
    bmp_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]

    # Iterate through BMP files and check for 0 or NaN values
    for bmp_file in bmp_files:
        bmp_path = os.path.join(folder_path, bmp_file)
        has_zero, has_nan = check_for_zero_nan_values(bmp_path)

        # Print image names with 0 or NaN values
        if has_zero or has_nan:
            print(f"Image: {bmp_file} - Zero values: {has_zero}, NaN values: {has_nan}")

# Replace 'path/to/your/image/folder' with the actual path to your image folder
image_folder_path = 'path/to/your/image/folder'
find_images_with_zero_nan(image_folder_path)
