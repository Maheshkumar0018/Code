import os
import json

def convert_to_yolov7_format(annotation, class_mapping):
    yolov7_lines = []

    for obj in annotation['objects']:
        label = obj['label']
        if label in class_mapping:
            class_index = class_mapping[label]
            bbox = obj['bbox']
            x_center = (bbox[0] + bbox[2]) / 2.0
            y_center = (bbox[1] + bbox[3]) / 2.0
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            yolov7_line = f"{class_index} {x_center} {y_center} {width} {height}"
            yolov7_lines.append(yolov7_line)

    return yolov7_lines

path_to_json = '/modified_labels/'
output_folder = '/yolov7_labels/'
class_mapping = {'fruit': 0, 'non-fruit': 1}  # Update with your specific class mapping

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
    input_path = os.path.join(path_to_json, file_name)
    output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.txt')

    with open(input_path) as json_file:
        data = json.load(json_file)
        yolov7_lines = convert_to_yolov7_format(data, class_mapping)

        with open(output_path, 'w') as output_file:
            output_file.write('\n'.join(yolov7_lines))

print("Conversion to YOLOv7 format completed.")
