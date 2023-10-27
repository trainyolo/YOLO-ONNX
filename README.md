# YOLOv8 ONNX Inference Library

Welcome to the YOLOv8 ONNX Inference Library, a lightweight and efficient solution for performing object detection with YOLOv8 using the ONNX runtime. This library is designed for cloud deployment and embedded devices, providing minimal dependencies and easy installation via pip.

## Installation

You can install the library using pip from the GitHub URL:

```bash
pip install git+https://github.com/trainyolo/YOLO-ONNX.git
```
## Getting Started

Here's a quick guide to get started with using the library for YOLOv8 object detection:

```python
from PIL import Image
from yolo_onnx.yolov8_onnx import YOLOv8

# initialize model
yolov8_detector = YOLOv8('./yolov8n.onnx')

# load image
img = Image.open('./test_image.jpg')

# do inference
detections = yolov8_detector(img, size=640, conf_thres=0.3, iou_thres=0.5)

# print results
print(detections)
```

## License
This library is distributed under the MIT License.

## Contact
If you have any questions or need assistance, feel free to contact us at info@trainyolo.com