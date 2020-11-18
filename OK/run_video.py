import torch
from lane_detect.erfnet import ERFNet
import cv2
import torch.nn.functional as F
import numpy as np

# LANE DETECT
PATH = "lane_detect/state_dict_model.pt"
img_width = 976
img_height = 208
H_offset = 320
LANE_THRESH = 100
COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

lane_detect_model = ERFNet(5)
lane_detect_model = torch.nn.DataParallel(
    lane_detect_model, device_ids=[0]).cuda()

weighted = torch.load(PATH)
lane_detect_model.load_state_dict(weighted, strict=False)
lane_detect_model.eval()


def image_feed(img):
    image = img.copy()[H_offset:, :, :]
    h, w = image.shape[:2]
    image = cv2.resize(image, (img_width, img_height))
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    image = image.unsqueeze_(0)
    return image, h, w


# OBJECT DETECT
cfgPath = "D:/github/thief_detection/demo/yolov4-tiny.cfg"
weightPath = "D:/github/thief_detection/demo/yolov4-tiny.weights"
classPath = "D:/github/thief_detection/demo/coco.names"
SCORE_THRESH = 0.5
NMI_THRESH = 0.3

net = cv2.dnn.readNet(weightPath, cfgPath)
object_detect_model = cv2.dnn_DetectionModel(net)
object_detect_model.setInputParams(size=(416, 416), scale=1/255.0, swapRB=True)
object_detect_class_names = open(classPath).read().strip().split("\n")

cap = cv2.VideoCapture("MOV_0540.mp4")


def lane_detection(image):
    imagez = image.copy()
    input_img, H_ori, W_ori = image_feed(imagez)
    input_var = torch.autograd.Variable(input_img, volatile=True)
    output, output_exist = lane_detect_model(input_var)
    output = F.softmax(output, dim=1)
    pred = output.data.cpu().numpy()
    pred_exist = output_exist.data.cpu().numpy()

    masked = np.zeros((H_ori, W_ori, 3))
    for num in range(4):
        prob_map = (pred[0][num+1]*255).astype(int)
        save_img = cv2.blur(prob_map, (9, 9))
        prob_map = cv2.resize(np.uint8(save_img), (W_ori, H_ori))
        masked[prob_map > LANE_THRESH] = COLOR[num]

    masked_full = np.zeros(imagez.shape, np.uint8)
    masked_full[H_offset:, :, :] = masked

    drawed = imagez.copy()
    dst = cv2.addWeighted(masked_full, 0.6, drawed, 0.9, 0)
    cv2.line(dst, (0, H_offset), (dst.shape[1], H_offset), (0, 255, 255), 1)

    return dst


def object_detection(image, img_drawed=None):
    drawed = image.copy()
    if img_drawed is not None:
        drawed = img_drawed

    classes, scores, boxes = object_detect_model.detect(
        image, SCORE_THRESH, NMI_THRESH)
    # COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    for (classid, score, box) in zip(classes, scores, boxes):
        # if classid[0] > 0 and classid[0] < 9:
        # color = COLORS[int(classid) % len(COLORS)]
        # label = "%s : %f" % (
        # object_detect_class_names[classid[0]], score)
        cv2.rectangle(drawed, box, (0, 255, 255), 2)
        # cv2.putText(drawed, label, (box[0], box[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return drawed


while 1:
    _, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    frame = frame[:700, :]
    if not _:
        break

    dst = lane_detection(frame)
    # dst = object_detection(frame, dst)
    cv2.imshow("dst", dst)
    cv2.waitKey(1)
