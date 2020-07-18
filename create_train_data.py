import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = r"C:\Users\chuen\Desktop\Python\image_recognition_basics\Python-Gesture-Recognition\extracted_images"

IMG_TYPES = ["fist", "okay", "palm", "peace"]

IMG_SIZE = 70

training_data = []

for type in IMG_TYPES:
    path = os.path.join(DATADIR, type)
    type_num = IMG_TYPES.index(type)

    for img in os.listdir(path):

        try:
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resize_img = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
            training_data.append([resize_img, type_num])
        except Exception as e:
            pass

random.shuffle(training_data)
# Time to pickle training data
X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

file_out = open("X.pickle", "wb")
pickle.dump(X, file_out)
file_out.close()

file_out = open("Y.pickle", "wb")
pickle.dump(Y, file_out)
file_out.close()
