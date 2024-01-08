import subprocess

def train_yolov7(workers, device, batch_size, data, img_size, cfg, weights, name, hyp):
    cmd = [
        'python', 'train.py',
        '--workers', str(workers),
        '--device', str(device),
        '--batch-size', str(batch_size),
        '--data', data,
        '--img', str(img_size), str(img_size),
        '--cfg', cfg,
        '--weights', weights,
        '--name', name,
        '--hyp', hyp
    ]

    subprocess.run(cmd)

# Example usage
train_yolov7(workers=8, device='cpu', batch_size=2, data='data/custom.yaml',
              img_size=640, cfg='cfg/training/yolov7.yaml', weights='./yolov7.pt', name='yolov7', 
              hyp='data/hyp.scratch.p5.yaml')

https://stackoverflow.com/questions/76541998/using-a-python-function-how-can-i-trigger-the-training-function-in-train-py


import subprocess
import os

def run_yolov7_train():
    try:
        # Specify the path to your YOLOv7 train.py script
        train_script_path = '/path/to/yolov7/train.py'

        # Build the command to run
        command = ["python", train_script_path, "--workers", "8", "--device", "0", "--batch-size", "32", "--data", "data/coco.yaml", "--img", "640 640", "--cfg", "cfg/training/yolov7.yaml", "--weights", "", "--name", "yolov7", "--hyp", "data/hyp.scratch.p5.yaml"]

        # Run the command and capture the output
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Extract the path to the trained weights from the runs folder
        runs_folder = os.path.join(os.getcwd(), 'runs')  # Assuming the runs folder is in the current working directory
        weight_file_path = os.path.join(runs_folder, 'yolov7', 'weights', 'best.pt')

        return weight_file_path

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

# Call the function to run YOLOv7 training script and get the weight file path
trained_weight_path = run_yolov7_train()

# Use the trained_weight_path variable as needed
if trained_weight_path:
    print(f"Path to trained weight file: {trained_weight_path}")
else:
    print("Training failed.")


--------------------------------------------------------------------------------------------
import os
import subprocess

def train_yolov7(workers, device, batch_size, data, img_size, cfg, weights, name, hyp):
    base_folder = name

    # Check if the base folder already exists
    if os.path.exists(f'runs/train/{base_folder}'):
        # Find the latest existing folder with the base name and increment the index
        index = 1
        while os.path.exists(f'runs/train/{base_folder}{index}'):
            index += 1

        # Create the new folder
        folder_name = f'{base_folder}{index}'
        os.makedirs(f'runs/train/{folder_name}')
    else:
        # Create the base folder if it doesn't exist
        folder_name = base_folder
        os.makedirs(f'runs/train/{folder_name}')

    # Construct the command for training
    cmd = [
        'python', 'train.py',
        '--workers', str(workers),
        '--device', str(device),
        '--batch-size', str(batch_size),
        '--data', data,
        '--img', str(img_size), str(img_size),
        '--cfg', cfg,
        '--weights', weights,
        '--name', folder_name,  # Use the folder name as the output name
        '--hyp', hyp
    ]

    # Run the YOLOv7 training script
    subprocess.run(cmd)

    # Return the path where the results are saved
    result_path = f'runs/train/{folder_name}'
    print(f'Training results saved in: {result_path}')
    return result_path

# Example usage
result_path_1 = train_yolov7(workers=8, device=0, batch_size=32, data='data/coco.yaml', img_size=640, cfg='cfg/training/yolov7.yaml', weights='', name='yolo', hyp='data/hyp.scratch.p5.yaml')
result_path_2 = train_yolov7(workers=8, device=0, batch_size=32, data='data/coco.yaml', img_size=640, cfg='cfg/training/yolov7.yaml', weights='', name='yolo', hyp='data/hyp.scratch.p5.yaml')

print(f'Results for the first run are saved in: {result_path_1}')
print(f'Results for the second run are saved in: {result_path_2}')
_-----------------------

import subprocess

def export_yolov7(device, weights, img_size, output_folder, include=['coreml', 'onnx', 'torchscript']):
    cmd = [
        'python', 'export.py',
        '--device', str(device),
        '--weights', weights,
        '--img-size', str(img_size),
        '--include', ','.join(include),
        '--dynamic',  # Add this if you want dynamic ONNX shape
        '--dynamic-export',  # Add this if you want dynamic ONNX shape
        '--optimize-export',  # Add this for optimized ONNX export
        '--dynamic-export',  # Add this for dynamic ONNX shape
        '--simplify',  # Add this to simplify exported ONNX model
        '--simplify-input-shape', '640', '640'  # Adjust input shape accordingly
    ]

    output_path = f'{output_folder}/yolov7.onnx'
    cmd.extend(['--output', output_path])

    subprocess.run(cmd)

# Example usage
export_yolov7(device='cpu', weights='./yolov7.pt', img_size=640, output_folder='./exported_models', include=['onnx'])






model_path = "runs/train/folder/weights/best.pt"

# Remove "best.pt" from the end of the path
model_path_without_best_pt = model_path.rsplit('/', 1)[0]

print(model_path_without_best_pt)
