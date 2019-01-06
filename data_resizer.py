import cv2
from PIL import Image

def resize(x):
    x_out = []

    for i in range(len(x)):
        img = cv2.cvtColor(x[i], cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img,dsize=(96,96))
        x_out.append(img)

    return np.array(x_out)

X_train_s = resize(x_train_s)
X_ref = resize(x_ref)
X_test_s = resize(x_test_s)
X_test_b = resize(x_test_b)
