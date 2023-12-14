import numpy as np
import torch
from PIL import Image
from utils.datasets import letterbox
from utils.plots import plot_one_box
import openvino as ov
import cv2
from typing import Tuple



def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv7 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize, converts color space from BGR (default in OpenCV) to RGB and changes data layout from HWC to CHW.

    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
      img0 (np.ndarray): original image
    """
    # resize
    img = letterbox(img0, auto=False)[0]

    # Convert
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0


def prepare_input_tensor(image: np.ndarray):
    """
    Converts preprocessed image to tensor format according to YOLOv7 input requirements.
    Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor object with float data in [0, 1] range

    Parameters:
      image (np.ndarray): image for conversion to tensor
    Returns:
      input_tensor (torch.Tensor): float tensor ready to use for YOLOv7 inference
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp16/32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)

    # Resize the input tensor to match the expected shape [1, 3, 1280, 1280]
    input_tensor = resize_input_tensor(input_tensor, (1280, 1280))
    print('#################################')
    print('input_tensor shape: ',input_tensor.shape)

    return input_tensor

def resize_input_tensor(input_tensor: np.ndarray, target_shape: Tuple[int, int]):
    """
    Resize the input tensor to the target shape.

    Parameters:
      input_tensor (np.ndarray): Input tensor
      target_shape (Tuple[int, int]): Target shape (height, width)
    Returns:
      resized_tensor (np.ndarray): Resized tensor
    """
    resized_tensor = np.zeros((input_tensor.shape[0], input_tensor.shape[1], target_shape[0], target_shape[1]), dtype=np.float32)

    for i in range(input_tensor.shape[0]):
        resized_tensor[i, 0] = cv2.resize(input_tensor[i, 0], target_shape, interpolation=cv2.INTER_LINEAR)
        resized_tensor[i, 1] = cv2.resize(input_tensor[i, 1], target_shape, interpolation=cv2.INTER_LINEAR)
        resized_tensor[i, 2] = cv2.resize(input_tensor[i, 2], target_shape, interpolation=cv2.INTER_LINEAR)
    print('##################################')
    print('resized_tensor shape: ',resized_tensor.shape)
    print('resized_tensor: ',input_tensor)
    return resized_tensor



# label names for visualization
#DEFAULT_NAMES = ['ambulance', 'auto-rickshaw', 'bicycle', 'bus', 'car', 'garbage van', 'human hauler', 'minibus',
#        'minivan', 'motorbike', 'Pickup', 'army vehicle', 'police car', 'rickshaw', 'scooter', 'Suv', 'taxi',
#        'three-wheelers (CNG)', 'truck', 'van', 'wheelbarrow']


DEFAULT_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
    'donut', 'cake', 'chair', 'couch', 'potted plant', 
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]





# obtain class names from model checkpoint
state_dict = torch.load("./yolov7-w6.pt", map_location="cpu")
if hasattr(state_dict["model"], "module"):
    NAMES = getattr(state_dict["model"].module, "names", DEFAULT_NAMES)
else:
    NAMES = getattr(state_dict["model"], "names", DEFAULT_NAMES)

del state_dict

# colors for visualization
COLORS = {name: [np.random.randint(0, 255) for _ in range(3)]
          for i, name in enumerate(NAMES)}

from typing import List, Tuple, Dict
from utils.general import scale_coords, non_max_suppression


def detect(model: ov.Model, image_path, conf_thres: float = 0.25, iou_thres: float = 0.45, classes: List[int] = None, agnostic_nms: bool = False):
    """
    OpenVINO YOLOv7 model inference function. Reads image, preprocess it, runs model inference and postprocess results using NMS.
    Parameters:
        model (Model): OpenVINO compiled model.
        image_path (Path): input image path.
        conf_thres (float, *optional*, 0.25): minimal accpeted confidence for object filtering
        iou_thres (float, *optional*, 0.45): minimal overlap score for remloving objects duplicates in NMS
        classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
    Returns:
       pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
       orig_img (np.ndarray): image before preprocessing, can be used for results visualization
       inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
    """
    output_blob = model.output(0)
    img = np.array(Image.open(image_path))
    preprocessed_img, orig_img = preprocess_image(img)
    input_tensor = prepare_input_tensor(preprocessed_img)
    predictions = torch.from_numpy(model(input_tensor)[output_blob])
    pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    print('#################################')
    print('prediction: ', pred)
    return pred, orig_img, input_tensor.shape


def draw_boxes(predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str], colors: Dict[str, int]):
    """
    Utility function for drawing predicted bounding boxes on image
    Parameters:
        predictions (np.ndarray): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
        image (np.ndarray): image for boxes visualization
        names (List[str]): list of names for each class in dataset
        colors (Dict[str, int]): mapping between class name and drawing color
    Returns:
        image (np.ndarray): box visualization result
    """
    if not len(predictions):
        return image
    # Rescale boxes from input size to original image size
    predictions[:, :4] = scale_coords(input_shape[2:], predictions[:, :4], image.shape).round()

    # Write results
    for *xyxy, conf, cls in reversed(predictions):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image, label=label, color=colors[names[int(cls)]], line_thickness=1)
    return image

core = ov.Core()
# read converted model
model = core.read_model('./model/yolov7_w6_quant.xml')

import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

# load model on CPU device
compiled_model = core.compile_model(model, device.value)
boxes, image, input_shape = detect(compiled_model, './inference/images/horses.jpg')
image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES, COLORS)
# visualize results
Image.fromarray(image_with_boxes)

# Convert the image with drawn boxes to PIL format
output_image = Image.fromarray(image_with_boxes)
# Save the image to a file
output_image.save('./output/result_image.jpg')

print("completd.....")
