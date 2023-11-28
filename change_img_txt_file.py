import os

def read_classes(class_file):
    with open(class_file, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

def map_to_new_classes(old_class):
    # Define your mapping logic here
    try:
        old_class = int(old_class)
    except ValueError:
        return 'unknow'  # or any value that represents an unknown class

    if old_class in [0, 1, 2, 3, 4, 5]:  # Replace with the actual class indices
        return 0
    elif old_class in [6, 7]:  # Replace with the actual class indices
        return 1
    else:
        return 'unknow'  # or any value that represents an unknown class

def modify_and_save_txt_files(input_folder, output_folder, class_file):
    classes = read_classes(class_file)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, 'r') as file:
                lines = [line.strip().split() for line in file]

            modified_lines = []
            for line in lines:
                old_class = line[0]
                new_class = map_to_new_classes(old_class)
                modified_line = f"{new_class} {' '.join(line[1:])}"
                modified_lines.append(modified_line)

            # Save the modified lines to the output folder
            with open(output_path, 'w') as file:
                file.write('\n'.join(modified_lines))

if __name__ == "__main__":
    input_folder = "./labels/"  # Replace with the actual input folder path
    output_folder = "./outputs"  # Replace with the actual output folder path
    class_file = "./labels/classes.txt"  # Replace with the actual class file path

    modify_and_save_txt_files(input_folder, output_folder, class_file)
