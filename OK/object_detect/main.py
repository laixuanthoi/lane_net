import cv2
import numpy as np
from model import Model
import time
cap = cv2.VideoCapture("z.mkv")

skip_frame = 1
skip_count = 1
blur_kernel = 3
dilate_kernel = 5
delta_thresh = 10
min_area = 100

avg = None


cfgPath = "D:/github/thief_detection/demo/yolov4-tiny.cfg"
weightPath = "D:/github/thief_detection/demo/yolov4-tiny.weights"
classPath = "D:/github/thief_detection/demo/coco.names"


def drawing(image, classes, scores, boxes):
    drawed = image.copy()
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    for (classid, score, box) in zip(classes, scores, boxes):
        if classid[0] > 0 and classid[0] < 9:
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (model.class_names[classid[0]], score)
            cv2.rectangle(drawed, box, color, 1)
            cv2.putText(drawed, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow("drawed image", drawed)


def moving_detection(frame):
    classes, scores, boxes = model.predict(
        frame, 0.3, 0.6)
    drawing(frame, classes, scores, boxes)


frame_count = 0

while True:
    _, frame = cap.read()

    if not _:
        break
    H, W = frame.shape[:2]
    frame = cv2.resize(frame, (640, 480))
    de = (frame)

    # cv2.imshow("avg", avg)
    # cv2.imshow("frame", frame)
    cv2.waitKey(1)
