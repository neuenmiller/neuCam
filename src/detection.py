import cv2
import time
from collections import OrderedDict

time_limit_car = 14400  # 4 hours
time_limit_human = 600  # 10 minutes

cap = cv2.VideoCapture(0)

car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

trackedCar = OrderedDict()
trackedHuman = OrderedDict()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.3, 4)
    humans = human_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in cars:
        roi = frame[y:y + h, x:x + w]

        height, width, channels = roi.shape
        blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        

    for (x, y, w, h) in humans:
        roi = frame[y:y + h, x:x + w]



    cv2.imshow('Car detection with OpenCV', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()