import os
import random
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain

def load_labels_from_annotation(annotation_file):
    with open(annotation_file, 'r') as f:
        labels = [int(line.strip().split()[0]) for line in f]
    return labels

def balance_dataset(data_folder, output_folder, test_size=0.2, val_size=0.1, random_state=42):
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "validation"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "validation"), exist_ok=True)

    # Get a list of image files and their corresponding annotation files in the data folder
    image_files = [f for f in os.listdir(data_folder) if f.endswith(".jpg")]
    annotation_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]

    # Load class labels from annotation files
    labels = [load_labels_from_annotation(os.path.join(data_folder, ann_file)) for ann_file in annotation_files]

    # Flatten the list of labels
    labels = list(chain.from_iterable(labels))

    # Count the occurrences of each class in the dataset
    class_counts = Counter(labels)

    # Determine the minimum number of samples per class based on stratified sampling
    min_samples_per_class = min(class_counts.values())

    # Resample the data for each class to achieve a balanced dataset
    for label, count in class_counts.items():
        image_files_by_label = [img_file for img_file, img_label in zip(image_files, labels) if img_label == label]
        resampled_image_files = random.sample(image_files_by_label, min_samples_per_class)

        # Split the resampled data into train, test, and validation sets
        X_train, X_test = train_test_split(resampled_image_files, test_size=test_size, random_state=random_state)
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


data_folder = './data'
output_folder = './balanced_data'
balance_dataset(data_folder, output_folder, test_size=0.2, val_size=0.1)
