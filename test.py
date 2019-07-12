import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import cv2

# Variables globales
TEST_DIR = './testDir'
TRAIN_DIR = './trainDir'
LEARNING_RATE = 1e-3
MODEL_NAME = "hapiness-{}-{}.model".format(LEARNING_RATE, "6convfire")
IMAGE_SIZE = 50

# Etiquetando información
def label_image(img):
    img_name = img.split(".")[-3]
    if img_name == "cat":
        return [1,0]
    elif img_name == "dog":
        return [0,1]

# Para entrenar los datos de entrada
def train_data_loader():
    training_data = []
    for img in tqdm(os.listdir(path=TRAIN_DIR)):
        img_label = label_image(img)
        path_to_img = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE),(IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([np.array(img), np.array(img_label)])
    shuffle(training_data)
    np.save("training_data_new.npy", training_data)
    return training_data

# Para probar los datos de entrada
def testing_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        img_labels = img.split(".")[0]
        path_to_img = os.path.join(TEST_DIR, img)
        img = cv2.resize(cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE),(IMAGE_SIZE, IMAGE_SIZE))
        test_data.append([np.array(img), np.array(img_labels)])

    shuffle(test_data)
    np.save("test_dataone.npy", test_data)
    return test_data

