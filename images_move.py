import os
import shutil

def copy_images(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                source_path = os.path.join(root, file)
                if file != 'bb_info.txt':
                    target_path = os.path.join(target_folder, file)
                    shutil.copy(source_path, target_path)

if __name__ == "__main__":
    source_folder = "./UECFOOD256"
    target_folder = "./images"
    copy_images(source_folder, target_folder)
