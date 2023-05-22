import numpy as np
import cv2
import time
import os
from grabscreen import grab_screen
from grabkeys import key_check
from PIL import ImageGrab
from PIL import Image

def keys_to_output(keys):

    # [A,W,D]
    output = [0,0,0,0]

    if 'A' in keys:
        output[0] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'S' in keys:
        output[3] = 1
    #else:
    #    output[4] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exist, loading previous data!')
    training_data = list(np.load(file_name))


else:
    print('File does not exist, starting fresh')
    training_data = []

i=0
def main():

    global training_data
    global i
    while (True):

        i = i+1
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 1440, 810)))
        #screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720)))

        #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.resize(screen, (480, 270))


        keys = key_check()

        output = keys_to_output(keys)

        # print(type(training_data))
        # training_data = list(training_data)
        # training_data.append([screen, output])



        img_2 = Image.fromarray(screen)  # NumPy array to PIL image
        os.makedirs('./w', exist_ok=True)
        os.makedirs('./a', exist_ok=True)
        os.makedirs('./d', exist_ok=True)
        os.makedirs('./s', exist_ok=True)
        os.makedirs('./n', exist_ok=True)
        if output == [1, 0, 0,0]:
            img_2.save(f'./a/a34_{i}.jpg','png')  # save PIL image
            time.sleep(0.5)
        elif output == [0, 1, 0,0]:
            img_2.save(f'./w/w34_{i}.jpg','png')
            time.sleep(0.5)
        elif output == [0, 0, 1,0]:
            img_2.save(f'./d/d34_{i}.jpg','png')
            time.sleep(0.5)
        elif output == [0, 0, 0,1]:
            img_2.save(f'./s/s34_{i}.jpg','png')
            time.sleep(0.5)
        #else:
        #    img_2.save(f'./n/n1_{i}.jpg','png')
        #    time.sleep(0.5)

        #dwd
        #
        # if len(training_data) % 2 == 0:
        #     print(training_data)
        #     training_data = np.array(training_data)#, dtype=object)
        #     print(len(training_data))
        #     np.save(file_name, training_data)
        #

main()