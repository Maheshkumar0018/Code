import os
import xml.etree.ElementTree as ET

def xml_to_yolo(xml_path, yolo_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image information
    size_elem = root.find('size')
    if size_elem is None:
        raise ValueError("Missing 'size' element in XML.")

    width_elem = size_elem.find('width')
    height_elem = size_elem.find('height')
    if width_elem is None or height_elem is None:
        raise ValueError("Missing 'width' or 'height' element in XML.")
    
    # Check if width or height values are empty or None
    if width_elem.text is None or not width_elem.text.strip():
        raise ValueError("Width value is empty or None in XML.")

    if height_elem.text is None or not height_elem.text.strip():
        raise ValueError("Height value is empty or None in XML.")
   
    image_width = int(width_elem.text)
    image_height = int(height_elem.text)

    # Define your class mapping (replace with your actual class labels and indices)
    class_mapping = {'cat': 0, 'dog': 1, 'bird': 2, 'person': 3}

    # Open/create YOLO annotation file
    with open(yolo_path, 'w') as yolo_file:
        # Process each object in the XML file
        for obj in root.findall('object'):
            class_label = obj.find('name').text

            # Check if the class is in the mapping
            if class_label in class_mapping:
                class_index = class_mapping[class_label]
            else:
                raise ValueError(f"Class '{class_label}' not found in the class mapping.")

            bbox = obj.find('bndbox')
            if bbox is None:
                raise ValueError("Missing 'bndbox' element in XML.")

            xmin_elem = bbox.find('xmin')
            ymin_elem = bbox.find('ymin')
            xmax_elem = bbox.find('xmax')
            ymax_elem = bbox.find('ymax')

            if xmin_elem is None or ymin_elem is None or xmax_elem is None or ymax_elem is None:
                raise ValueError("Missing bounding box coordinates in XML.")

            xmin = int(xmin_elem.text)
            ymin = int(ymin_elem.text)
            xmax = int(xmax_elem.text)
            ymax = int(ymax_elem.text)

            # Convert coordinates to YOLO format
            x_center = (2 * xmin + (xmax - xmin)) / (2.0 * image_width)
            y_center = (2 * ymin + (ymax - ymin)) / (2.0 * image_height)
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Write to YOLO annotation file
            yolo_file.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_index, x_center, y_center, width, height))

def convert_folder(xml_folder, yolo_folder):
    # Create YOLO folder if it doesn't exist
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)

    # Process each XML file in the folder
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            yolo_path = os.path.join(yolo_folder, xml_file.replace(".xml", ".txt"))
            xml_to_yolo(xml_path, yolo_path)

if __name__ == "__main__":
    # Replace 'xml_folder' and 'yolo_folder' with your actual folder paths
    xml_folder = "./xml/"
    yolo_folder = "./txt_annt/"

    convert_folder(xml_folder, yolo_folder)
