import cv2
import matplotlib.pyplot as plt

config_file = 'C:\\Users\\sroya\\Downloads\\ObjectDetectionUsingAi\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'C:\\Users\\sroya\\Downloads\\ObjectDetectionUsingAi\\frozen_inference_graph.pb'

model = cv2.dnn.DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'C:\\Users\\sroya\\Downloads\\ObjectDetectionUsingAi\\LABELS.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

img = cv2.imread('C:\\Users\\sroya\\Downloads\\object detection materials\\man in traffic.png')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

ClassIndex , confidence , bbox = model.detect(img,confThreshold=0.6)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd , conf , boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    # Draw the bounding box
    cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[0]+boxes[2], boxes[1]+boxes[3]), (255, 0, 0), 2)
    # Put the class label and confidence score
    text = f"{classLabels[ClassInd-1]}: {conf:.2f}"
    cv2.putText(img, text, (boxes[0], boxes[1]-10), font ,fontScale=font_scale, color=(0,255,0),thickness=3)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
