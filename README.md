# Object-detection-using-web-camera

## object_detection_1
This project uses the YOLOv4 (You Only Look Once) model for real-time object detection via a webcam
using OpenCV and Python.

## Requirement
Make sure you have the following installed: Python 3.x OpenCV (opencv-python) NumPy YOLOv4 weights
and configuration files coco.names (list of class labels)

## Algorithm
Step 1: Load YOLOv4 configuration, weights, and class names (yolov4.cfg, yolov4.weights, coco.names).
Step 2: Initialize the webcam using OpenCV and capture frames in real time.
Step 3: Convert each frame to a blob and feed it to the YOLOv4 network for object detection.
Step 4: Process the network output to extract class IDs, confidence scores, and bounding boxes for
detected objects.
Step 5: Draw bounding boxes and labels on the frame, display the output, and exit when the user presses
q.

## program
```
import cv2
import numpy as np
# Load YOLOv4 model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = open("coco.names").read().strip().split("\n")
# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
ret, frame = cap.read()
if not ret:
break
height, width, _ = frame.shape
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
detections = net.forward(output_layers)
for output in detections:
for detection in output:
scores = detection[5:]
class_id = np.argmax(scores)
confidence = scores[class_id]
if confidence > 0.5:
center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, h
x, y = center_x - w // 2, center_y - h // 2
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(frame, f"{classes[class_id]} {confidence:.2f}", (x, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# Display output
cv2.imshow("YOLOv4 Real-Time Detection", frame)
# Press 'q' to exit
if cv2.waitKey(1) & 0xFF == ord("q"):
break
cap.release()
cv2.destroyAllWindows()
```
## output
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/b4cb23d9-564a-44b7-a92e-3a290063c505" />

