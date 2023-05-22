import time
import numpy as np
import torch
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms


def cam_recog():
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    # labels = {0: 'Chilsung', 1: 'Coca', 2: 'Letsbe', 3: 'Milkis', 4: 'Mountain Dew', 5: 'Welchs'}
    labels = {0: '칠성 사이다', 1: '코카콜라', 2: '레쓰비', 3: '밀키스', 4: '마운틴 듀', 5: '웰치스 포도'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = models.vgg19_bn(pretrained=False)
    # model.classifier[6] = nn.Linear(in_features=4096, out_features=3)
    #model = models.mobilenet_v3_large(pretrained=False)
    #model.classifier[3] = nn.Linear(in_features=1280, out_features=3)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)

    model.load_state_dict(torch.load("./models/ResNet50_20.pt", map_location=device))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        time_start = time.time()
        while True:
            ret, frame = cap.read()
            # 왜 RGB 변환을 2번 해야 되는걸까?
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            pil_img = Image.fromarray(frame)
                                        # reshape() 보다 언스퀴즈 하는게 코드 수정이 덜함
            input_img = data_transforms(pil_img).unsqueeze(0).to(device)
            out = model(input_img)
            softmax_result = F.softmax(out)
            top1_prob, top1_label = torch.topk(softmax_result, 1)
            print(top1_prob, labels.get(int(top1_label)))
            acc = ">>  " + str(round(top1_prob.item() * 100, 2)) + "%"
            Accuracy = round(top1_prob.item() * 100, 2)
            Label = labels.get(int(top1_label))

            b, g, r, a = 0, 255, 0, 0
            fontpath = 'font/gulim.ttc'
            font = ImageFont.truetype(fontpath, 20)
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 100), f'{labels.get(int(top1_label))}', font=font, fill=(b, g, r, a))
            draw.text((30, 160), f'{acc}', font=font, fill=(b, g, r, a))
            frame = np.array(pil_img)

            cv2.imshow("TEST", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif (time.time() - time_start > 7) & (round(top1_prob.item() * 100, 2)>=85):
                break
        cap.release()
        cv2.destroyAllWindows()
    return Accuracy, Label


if __name__ == '__main__':
    a, l = cam_recog()
    print(a, l)

