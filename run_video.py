import torch
from torchsummary import summary
from torchvision import transforms
import models
import cv2
import torch.nn.functional as F
import numpy as np
import dataset as ds
import utils.transforms as tf


# PATH = "state_dict_model.pt"
PATH = "trained/ERFNet_trained.tar"
num_class = 5 #Culane
img_width = 976
img_height = 208
H_offset = 360
LANE_THRESH = 110
image_path = "D:/github/dataset/driver_23_30frame/05151640_0419.MP4/00000.jpg"


def image_feed(img):
    image = img.copy()[H_offset:, :, :]
    h,w = image.shape[:2]
    image = cv2.resize(image, (img_width, img_height))
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    # image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze_(0)
    return image, h, w

model = models.ERFNet(5)
checkpoint = torch.load(PATH)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

cap = cv2.VideoCapture("2.mkv")
while 1:
    _, frame = cap.read()
    frame= cv2.resize(frame, (1280,720))
    if not _:
        break
    
    # imagez = cv2.imread(frame)
    imagez = frame
    input_img, H_ori, W_ori = image_feed(imagez)

    input_var = torch.autograd.Variable(input_img, volatile=True)
    output, output_exist = model(input_var)
    output = F.softmax(output, dim=1)
    pred = output.data.cpu().numpy()
    pred_exist = output_exist.data.cpu().numpy()


    # cv2.imshow("imagez", imagez)

    COLOR = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]

    masked = np.zeros((H_ori, W_ori, 3))
    # masked[:] = (255,255,255)
    for num in range(4):
        prob_map = (pred[0][num+1]*255).astype(int)
        save_img = cv2.blur(prob_map,(9,9))
        prob_map = cv2.resize(np.uint8(save_img), (W_ori, H_ori))
        masked[prob_map >LANE_THRESH] = COLOR[num]

    masked_full = np.zeros(imagez.shape, np.uint8)
    masked_full[H_offset:,:,:] = masked

    drawded = imagez.copy()
    # print(drawded.dtype, masked_full.dtype)
    # drawded[masked_full != 0] = masked_full
    dst = cv2.addWeighted(masked_full,0.6,drawded,0.9,0)

    # cv2.imshow("masked_full", masked_full)
    cv2.imshow("dst", dst)
        # cv2.imshow("prob_map", np.uint8(prob_map))
    cv2.waitKey(1)