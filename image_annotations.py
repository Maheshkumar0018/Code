import os
import random
import shutil
import cv2

def read_bb_info_files(root_folder):
    all_annotations = []
    category_mapping = {}

    category_file_path = os.path.join(root_folder, "category.txt").replace("\\", "/")
    if os.path.exists(category_file_path):
        with open(category_file_path, "r") as category_file:
            header_line = next(category_file)  # Skip the header line
            for line in category_file:
                folder_name, label = line.strip().split("\t")
                category_id = int(folder_name)
                category_mapping[folder_name] = category_id - 1  # Apply zero-based indexing

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            bb_info_file_path = os.path.join(folder_path, "bb_info.txt")
            if os.path.exists(bb_info_file_path):
                with open(bb_info_file_path, "r") as file:
                    first_line = next(file, None)
                    if first_line and first_line.strip() == "img x1 y1 x2 y2":
                        pass

                    for line_num, line in enumerate(file, start=1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            image_name, x1, y1, x2, y2 = line.split()
                            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                            image_name = os.path.join(folder_path, image_name)
                            all_annotations.append((image_name, x1, y1, x2, y2, folder_name))
                        except ValueError:
                            print(f"Invalid data in {bb_info_file_path}, line {line_num}: {line}")
                            continue

    return all_annotations, category_mapping

def create_annotation_files(annotations_list, category_mapping):
    image_annotation_folder = os.path.join(os.getcwd(), "image_annotation")
    os.makedirs(image_annotation_folder, exist_ok=True)

    for annotation in annotations_list:
        image_name, x1, y1, x2, y2, folder_name = annotation

        # Use folder_name as the category_id directly
        category_id = category_mapping[folder_name]

        # Read the image using OpenCV to get its height and width
        image_name = image_name + ".jpg"
        image = cv2.imread(image_name)
        if image is None:
            print(f"Unable to read image: {image_name}")
            continue

        image_height, image_width, _ = image.shape

        # Calculate YOLO format bounding box coordinates
        x_center = (x1 + x2) / (2 * image_width)
        y_center = (y1 + y2) / (2 * image_height)
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height

        # Create a new file with the image name as the filename
        image_num = os.path.splitext(os.path.basename(image_name))[0]  # Remove the file extension
        new_file_name = f"{image_num}.txt"
        new_file_path = os.path.join(image_annotation_folder, new_file_name)

        # Write the annotation to the new file in YOLO format
        with open(new_file_path, "w") as new_file:
            new_file.write(f"{category_id} {x_center} {y_center} {w} {h}")


root_folder = './UECFOOD256' 


annotations_list, category_mapping = read_bb_info_files(root_folder)

create_annotation_files(annotations_list, category_mapping)

