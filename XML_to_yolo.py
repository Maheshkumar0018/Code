import os
import xml.etree.ElementTree as ET

# Input and output folders
xml_folder = 'path/to/xml_folder'
output_folder = 'path/to/output_folder'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through XML files in the input folder
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)

        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Create YOLO format text
        yolo_txt = ''
        for obj in root.findall('.//object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # YOLO format: class x_center y_center width height
            width = xmax - xmin
            height = ymax - ymin
            x_center = xmin + width / 2
            y_center = ymin + height / 2

            yolo_txt += f"{class_name} {x_center} {y_center} {width} {height}\n"

        # Save YOLO format text to a file in the output folder
        output_txt_path = os.path.join(output_folder, os.path.splitext(xml_file)[0] + '.txt')
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write(yolo_txt)

print("Conversion complete.")
