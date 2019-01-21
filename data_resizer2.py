import cv2
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img

img_path = 'pictures/'
NO = 1

def resize(x):
    x_out = []

    for i in range(len(x)):
        img = cv2.resize(x[i],dsize=(96,96))
        x_out.append(img)

    return np.array(x_out)

x = []

while True:
    if not os.path.exists(img_path + str(NO) + ".jpg"):
        break
    img = Image.open(img_path + str(NO) + ".jpg")
    img = image.img_to_array(img)
    x.append(img)
    NO += 1

x_train = resize(x)
