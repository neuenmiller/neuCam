import cv2
import time
from collections import OrderedDict

time_limit = 14400  # 4 hours

cap = cv2.VideoCapture(0)

human_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

trackedCar = OrderedDict()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = human_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in cars:
        car_id = hash((x, y, w, h))
        current_time = time.time()
        if car_id not in trackedCar:
            trackedCar[car_id] = {'entry_time': current_time, 'coordinates': (x, y, w, h)}
        else:
            elapsed_time = current_time - trackedCar[car_id]['entry_time']
            if elapsed_time > time_limit:
                elapsed_time_hours = elapsed_time / 3600
                print(f"Car {car_id} has been parked for {elapsed_time_hours} hour(s)")

        cv2.putText(frame, f"Total cars: {len(trackedCar)}, ID: {car_id}, Time: {elapsed_time_hours}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.imshow('Car detection with OpenCV', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()


