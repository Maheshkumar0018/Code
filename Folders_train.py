import os
import random
from shutil import copyfile

def create_yolo_folders(source_images_folder, source_labels_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1.0"

    random.seed(seed)

    # Create output folders
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

    # Get lists of image and label files
    image_files = [f for f in os.listdir(source_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(source_labels_folder) if f.lower().endswith('.txt')]

    # Shuffle files randomly
    random.shuffle(image_files)

    # Calculate split indices
    total_files = len(image_files)
    train_split = int(train_ratio * total_files)
    val_split = int(val_ratio * total_files)

    # Copy files to respective folders
    for i, image_file in enumerate(image_files):
        source_image_path = os.path.join(source_images_folder, image_file)
        source_label_path = os.path.join(source_labels_folder, os.path.splitext(image_file)[0] + '.txt')

        if i < train_split:
            destination_folder = train_folder
        elif i < train_split + val_split:
            destination_folder = val_folder
        else:
            destination_folder = test_folder

        destination_image_path = os.path.join(destination_folder, 'images', image_file)
        destination_label_path = os.path.join(destination_folder, 'labels', os.path.splitext(image_file)[0] + '.txt')

        copyfile(source_image_path, destination_image_path)
        copyfile(source_label_path, destination_label_path)

    print("Folder structure created successfully.")

# Example usage:
source_images_folder = 'path/to/source/images'
source_labels_folder = 'path/to/source/labels'
output_folder = 'path/to/output/folder'

create_yolo_folders(source_images_folder, source_labels_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
