from torchvision import models
import torch
import torch.nn as nn
from PIL import ImageGrab
import cv2
import torch.nn.functional as F
import albumentations as Al
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from input_keys import PressKey, ReleaseKey
import time

labels = {0: 'a', 1: 'w', 2: 'd', 3: 's'}
    

def ingame_predic():
    test_transform = Al.Compose(
        [
            # A.SmallestMaxSize(max_size=160),
            Al.Resize(width = 480, height = 270),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = models.mobilenet_v3_large(pretrained=True)    # mobilenet-v3
    # net.classifier[3] = nn.Linear(in_features = 1280, out_features=4)


    #net = models.efficientnet_b4(pretrained=True)   # efficientnet
    #net.classifier[1] = nn.Linear(in_features=1792,out_features=4)  
    #net.load_state_dict(torch.load('./models/test2model_efv4.pt', map_location=device))
        
    net = models.mobilenet_v3_large(pretrained=True)   # efficientnet
    net.classifier[3] = nn.Linear(in_features=1280, out_features=4)  
    net.load_state_dict(torch.load('models\MV3model.pt', map_location=device))

    
    net.to(device)
    net.eval()

    while(True):
        with torch.no_grad():
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720))) # 1024, 768 화면을 받아서 Numpy Array로 전환
            # screen = cv2.imread('./test_image2.jpg') # test image
            # input_image = Image.fromarray(screen)
            input_image = test_transform(image=screen)['image'].float().unsqueeze(0).to(device)

            output = net(input_image)        
            softmax_result = F.softmax(output)
            top_prob, top_label = torch.topk(softmax_result, 1)
            prob = round(top_prob.item() * 100, 2)
            label = labels.get(int(top_label))
            # print(f'prob: {prob}, label: {label}')

            W = 0x11
            A = 0x1E
            S = 0x1F
            D = 0x20

            if (45 < prob) and (label == 'a'):
                PressKey(W)
                PressKey(A)
                time.sleep(1)
                ReleaseKey(W)
                ReleaseKey(A)

            elif (45 < prob) and (label == 'w'):
                PressKey(W)
                time.sleep(1)
                ReleaseKey(W)

            elif (45 < prob) and (label == 'd'):
                PressKey(D)
                PressKey(W)
                time.sleep(1)
                ReleaseKey(D)
                ReleaseKey(W)

            elif (45 < prob) and (label == 's'):
                PressKey(S)
                time.sleep(1)
                ReleaseKey(S)

            else:
                time.sleep(1)

        print(prob, label)
    return prob, label


if __name__ == '__main__':
    predic_prob, predic_label = ingame_predic()