import cv2
import numpy as np
from win32api import GetSystemMetrics
width,height=GetSystemMetrics(0),GetSystemMetrics(1)
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Name custom object
classes = []
with open("coco.name", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
lst=[r'faces_video.mp4',r'854097-hd_1920_1080_25fps.mp4',r'3253272-uhd_3840_2160_25fps.mp4',r'4265036-uhd_3840_2160_30fps.mp4']
# Load video file
video_capture = cv2.VideoCapture(lst[3])

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2 and class_id == 0:  # 0 index is for person class in COCO dataset
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[0] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Face {i + 1}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(frame, label, (x, y + 30), font, 3, (0, 255, 0), 2)
    frames=cv2.resize(frame,(1000,500))
    cv2.imshow("Face Detection using YOLO", frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
