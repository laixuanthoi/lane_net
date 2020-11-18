
import numpy as np
import time
import cv2


class Model:
    def __init__(self, configPath, weightPath, classPath, input_size=(416, 416)):
        self.model_input_size = input_size
        self.fps = 0
        self.loadClassNames(classPath)
        self.loadModel(configPath, weightPath)

    def loadClassNames(self, classPath):
        self.class_names = open(classPath).read().strip().split("\n")

    def loadModel(self, configPath, weightPath):
        self.net = cv2.dnn.readNet(weightPath, configPath)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(
            size=self.model_input_size, scale=1/255.0, swapRB=True)

    def predict(self, image, confidence_threshold, nms_threshold):
        start = time.time()
        classes, scores, boxes = self.model.detect(
            image, confidence_threshold, nms_threshold)
        end = time.time()
        print("Predicted in: {}".format(end - start))
        self.fps = 1/(end - start)
        return classes, scores, boxes

    def drawing(self, image, classes, scores, boxes):
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
