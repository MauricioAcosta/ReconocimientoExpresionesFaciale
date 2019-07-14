import matplotlib.pyplot as plt

import numpy as np
import tflearn
from red import getNetwork
from const import TEST_DIR, TRAIN_DIR, LEARNING_RATE, MODEL_NAME, IMAGE_SIZE

model = tflearn.DNN(getNetwork(), tensorboard_dir='log')
model.load(MODEL_NAME)

test_data = np.load("test_dataone.npy", allow_pickle=True)

figs = plt.figure()
for num, data in enumerate(test_data[0:12]):
    test_img = data[0]
    test_lable = data[1]
    test_img_feed = test_img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    t = figs.add_subplot(3, 4, num + 1)
    ores = test_img
    print('test_img_feed.shape', test_img_feed.shape)
    model_pred = model.predict([test_img_feed])
    print('\nmodel_pred', model_pred[0])
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
    t.imshow(ores, cmap="gray")
    emotion = np.array2string(test_lable)
    emotion = emotion.split('_')[0]
    plt.title("pred: " + pred_val + " real: " + emotion)

plt.show()