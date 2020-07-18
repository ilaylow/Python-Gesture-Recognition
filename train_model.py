import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

print(tf.__version__)

X = pickle.load(open(r"C:\Users\chuen\Desktop\Python\image_recognition_basics\Python-Gesture-Recognition\X.pickle", "rb"))


Y = pickle.load(open(r"C:\Users\chuen\Desktop\Python\image_recognition_basics\Python-Gesture-Recognition\Y.pickle", "rb"))

Y = tf.keras.utils.to_categorical(Y, num_classes = 4)


X = X / 255.0  # Normalise it

model = Sequential()

# Two Conv Layers
model.add(Conv2D(128, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))  #Using rectified linear unit
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))  #Using rectified linear unit
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

# Dense Layers - Fully Connected
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

model.fit(X, np.array(Y), batch_size = 10, epochs = 10, validation_split = 0.1)

model.save('128x2-CNN-GestureRecog.model')
