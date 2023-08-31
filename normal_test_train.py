import os
import random
import shutil
from sklearn.model_selection import train_test_split

def create_folder_structure(data_folder, output_folder, test_size=0.2, val_size=0.1, random_state=42):
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "validation"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "validation"), exist_ok=True)

    # Get a list of image files in the data folder
    image_files = [f for f in os.listdir(data_folder) if f.endswith(".jpg")]

    # Split the data into train, test, and validation sets
    X_train, X_test = train_test_split(image_files, test_size=test_size, random_state=random_state)
    X_train, X_val = train_test_split(X_train, test_size=val_size, random_state=random_state)

    # Move images and labels to their respective folders
    for image_file in X_train:
        shutil.copy(os.path.join(data_folder, image_file), os.path.join(output_folder, "images", "train", image_file))
        label_file = image_file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(data_folder, label_file), os.path.join(output_folder, "labels", "train", label_file))

    for image_file in X_test:
        shutil.copy(os.path.join(data_folder, image_file), os.path.join(output_folder, "images", "test", image_file))
        label_file = image_file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(data_folder, label_file), os.path.join(output_folder, "labels", "test", label_file))

    for image_file in X_val:
        shutil.copy(os.path.join(data_folder, image_file), os.path.join(output_folder, "images", "validation", image_file))
        label_file = image_file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(data_folder, label_file), os.path.join(output_folder, "labels", "validation", label_file))

# Example usage
data_folder = './images'
output_folder = './data'
create_folder_structure(data_folder, output_folder, test_size=0.2, val_size=0.1)
