import cv2
import numpy as np
import tflearn
import matplotlib.pyplot as plt

from red import getNetwork
from const import TEST_DIR, TRAIN_DIR, LEARNING_RATE, MODEL_NAME, IMAGE_SIZE

model = tflearn.DNN(getNetwork(), tensorboard_dir='log')
model.load(MODEL_NAME)

figs = plt.figure()
t = figs.add_subplot(1, 1, 1)

def predict(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('1 img_gray.shape', img_gray.shape)
    img_gray = cv2.resize(img_gray, (IMAGE_SIZE, IMAGE_SIZE))
    #t.imshow(img_gray, cmap="gray")
    print('before img_gray.shape', img_gray.shape)
    img_gray = img_gray.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    print('after img_gray.shape', img_gray.shape)
    model_pred = model.predict([img_gray])
    print('model_pred', model_pred)
    maximo = np.argmax(model_pred[0])
    print('maximo', maximo)
    if maximo == 0:
        pred_val = "happy"
    elif maximo == 1:
        pred_val = "sad"
    elif maximo == 2:
        pred_val = "surprised"
    else:
        print('no pude predecirlo', model_pred)
    print('pred_val', pred_val)
    return pred_val
    #plt.show()
    #exit(0)
