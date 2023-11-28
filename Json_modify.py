import os
import json

path_to_json = '/lala/'
output_folder = '/modified_labels/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
    input_path = os.path.join(path_to_json, file_name)
    output_path = os.path.join(output_folder, file_name)

    with open(input_path) as json_file:
        data = json.load(json_file)

        # Replace labels
        for annotation in data['annotations']:
            for obj in annotation['objects']:
                if obj['label'] in ['apple', 'banana', 'orange', 'grapes']:
                    obj['label'] = 'fruit'
                elif obj['label'] == 'human':
                    obj['label'] = 'non-fruit'

        # Save the modified data to a new JSON file in the output folder
        with open(output_path, 'w') as output_file:
            json.dump(data, output_file, indent=2)

print("Label replacement completed.")
