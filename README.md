# Object Detection Using Traffic Images

## Overview
This project implements object detection on traffic images using OpenCV's DNN module with an SSD MobileNet model. It detects objects such as pedestrians, vehicles, and traffic signs from an image and draws bounding boxes with confidence scores.

## Features
- Uses **SSD MobileNet V3** for object detection  
- Reads class labels from a file (`LABELS.txt`)  
- Detects objects in traffic images with bounding boxes and confidence scores  
- Utilizes OpenCV for processing and Matplotlib for visualization  

## Requirements
- Python 3.x  
- OpenCV (`cv2`)  
- Matplotlib  

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ObjectDetectionTraffic.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ObjectDetectionTraffic
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python matplotlib
   ```

## Usage
1. Ensure you have the following files:
   - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` (Model Configuration)  
   - `frozen_inference_graph.pb` (Trained Model Weights)  
   - `LABELS.txt` (Class Labels)  
   - Input traffic image (e.g., `man in traffic.png`)  

2. Run the detection script:
   ```bash
   python detect.py
   ```

## Code Explanation
- **Load the model**:  
  ```python
  model = cv2.dnn.DetectionModel(frozen_model, config_file)
  ```
- **Read class labels** from `LABELS.txt`:
  ```python
  with open(file_name, 'rt') as fpt:
      classLabels = fpt.read().rstrip('\n').split('\n')
  ```
- **Preprocess image for the model**:
  ```python
  model.setInputSize(320,320)
  model.setInputScale(1.0/127.5)
  model.setInputMean((127.5,127.5,127.5))
  model.setInputSwapRB(True)
  ```
- **Perform object detection**:
  ```python
  ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.6)
  ```
- **Draw bounding boxes and labels**:
  ```python
  for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
      cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[0]+boxes[2], boxes[1]+boxes[3]), (255, 0, 0), 2)
      text = f"{classLabels[ClassInd-1]}: {conf:.2f}"
      cv2.putText(img, text, (boxes[0], boxes[1]-10), font, fontScale=font_scale, color=(0,255,0), thickness=3)
  ```
- **Display the output image**:
  ```python
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()
  ```

## Sample Output
The model will detect objects in the image and display them with bounding boxes and labels.
