import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import cv2
from tqdm import tqdm
from const import TEST_DIR, TRAIN_DIR, LEARNING_RATE, MODEL_NAME, IMAGE_SIZE

# Etiquetando información
def label_image(img):
    img_name = img.split("_")[0]
    if img_name == "happy":
        return [1,0,0]
    elif img_name == "sad":
        return [0,1,0]
    elif img_name == "surprised":
        return [0,0,1]

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
train_data_loader()

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
testing_data()