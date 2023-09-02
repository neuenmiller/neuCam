import cv2
import time
import numpy as np
from collections import OrderedDict

def detect_and_label(roi, net, output_layers, object_type):
    height, width, channels = roi.shape
    blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id in [0, 2]:
                    label = f"{object_type}: {confidence}"
                    return True, label
    return False, None
                    

time_limit_car = 14400  # 4 hours
time_limit_human = 600  # 10 minutes

cap = cv2.VideoCapture(0)

car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

yolo_net = cv2.dnn.readNet("yolo_files/yolov3-tiny.weights", "yolo_files/yolov3-tiny.cfg")
layers_names = yolo_net.getLayerNames()
output_layers = [layers_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

trackedCar = OrderedDict()
trackedHuman = OrderedDict()

while True:
    current_time = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.3, 4)
    humans = human_cascade.detectMultiScale(gray, 1.3, 4)

#CAR DETECTION
    for (x, y, w, h) in cars:
        roi = frame[y:y + h, x:x + w]

        detected, label = detect_and_label(roi, yolo_net, output_layers, "Car")
        if detected:
            key = (x, y, w, h)
            if key not in trackedCar:
                trackedCar[key] = current_time

            over_limit_text = ""
            if current_time - trackedCar[key] > time_limit_car:
                over_limit_text = "- OVER TIME LIMIT"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label + over_limit_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

#HUMAN DETECTION
    for (x, y, w, h) in humans:
        roi = frame[y:y + h, x:x + w]

        detected, label = detect_and_label(roi, yolo_net, output_layers, "Car")
        if detected:
            key = (x, y, w, h)
            if key not in trackedHuman:
                trackedHuman[key] = current_time
                
            over_limit_text = ""
            if current_time - trackedHuman[key] > time_limit_human:
                over_limit_text = "- OVER TIME LIMIT"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label + over_limit_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow('Car and human detection with OpenCV and YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()