import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

def main():
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    cap = cv2.VideoCapture('veh2.mp4')

    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().splitlines()

    tracker = Tracker()

    cy1, cy2 = 322, 368
    offset = 6

    vh_down = {}
    counter_down = []

    vh_up = {}
    counter_up = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        list_of_rects = []
        for index, row in px.iterrows():
            x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
            c = class_list[d]
            if 'car' in c:
                list_of_rects.append([x1, y1, x2, y2])

        bbox_id = tracker.update(list_of_rects)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            if cy1 < (cy + offset) < cy1 + 10:
                vh_down[id] = time.time()
            if id in vh_down and cy2 < (cy + offset) < cy2 + 10:
                elapsed_time = time.time() - vh_down[id]
                if id not in counter_down:
                    counter_down.append(id)
                    speed = 10 / elapsed_time * 3.6  # Speed in km/h
                    cv2.putText(frame, f'{int(speed)} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            if cy2 < (cy + offset) < cy2 + 10:
                vh_up[id] = time.time()
            if id in vh_up and cy1 < (cy + offset) < cy1 + 10:
                elapsed1_time = time.time() - vh_up[id]
                if id not in counter_up:
                    counter_up.append(id)
                    speed1 = 10 / elapsed1_time * 3.6  # Speed in km/h
                    cv2.putText(frame, f'{int(speed1)} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
        cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
        cv2.putText(frame, f'Going Down: {len(counter_down)}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Going Up: {len(counter_up)}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


