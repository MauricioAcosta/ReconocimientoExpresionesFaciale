import os
import numpy as np

import tflearn
from red import getNetwork

import tensorflow as tf
tf.reset_default_graph()
from const import TEST_DIR, TRAIN_DIR, LEARNING_RATE, MODEL_NAME, IMAGE_SIZE

train_data_g = np.load('training_data_new.npy', allow_pickle=True)

model = tflearn.DNN(getNetwork(), tensorboard_dir='log')

if os.path.exists("{}.meta".format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model Loaded")

train = train_data_g[0:247]
test = train_data_g[247:]

#This is our Training data
X = np.array([i[0] for i in train]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
Y = [i[1] for i in train]

#This is our Training data
test_x = np.array([i[0] for i in test]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
test_y = [i[1] for i in test]


#model.fit(X, Y, n_epoch=6, validation_set=(test_x,  test_y), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.fit(X, Y, n_epoch=30, validation_set=(test_x,  test_y), snapshot_step=50, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

#model_pred = model.predict([test_x[0]])
#print('model_pred', model_pred)

