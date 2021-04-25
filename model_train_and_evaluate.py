import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2


import os
currdir = os.path.abspath(os.getcwd())


images_path = currdir + '/real_and_fake_face'


# CONSTANTS
FOLDERS = ["training_real", "training_fake"]
IMG_SIZE=32
TRAINING_PROPORTION = 0.8
VALIDATION_PROPORTION = 0
MAX_USE = 1
EPOCHS = 12
RUN_NAME = "Run D3 with IMG_SIZE=" + str(IMG_SIZE) + "EPOCHS=" + str(EPOCHS)


# Functions to Load Data

def loadTrainingData():
  X = []
  Y = []

  for folder in FOLDERS:
    path = os.path.join(images_path, folder)
    class_num = FOLDERS.index(folder)
    max_index = TRAINING_PROPORTION*len(os.listdir(path)) *MAX_USE * (1 - VALIDATION_PROPORTION)

    for img in (os.listdir(path))[:int(max_index)]:
      try:
        img_array = cv2.imread(os.path.join(path,img))
        new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        new_array = np.array(new_array)

        new_array = new_array.astype('float32')
        new_array /= 255

        X.append(new_array)
        Y.append(class_num)
      except Exception as e:
        continue
    
  return X, Y


def loadValidationData():
  X = []
  Y = []

  for folder in FOLDERS:
    path = os.path.join(images_path, folder)
    class_num = FOLDERS.index(folder)
    min_index = TRAINING_PROPORTION*len(os.listdir(path)) *MAX_USE * (1 - VALIDATION_PROPORTION)
    max_index = TRAINING_PROPORTION*len(os.listdir(path)) *MAX_USE

    for img in (os.listdir(path))[int(min_index):int(max_index)]:
      try:
        img_array = cv2.imread(os.path.join(path,img))
        new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        new_array = np.array(new_array)
        new_array = new_array.astype('float32')
        new_array /= 255

        X.append(new_array)
        Y.append(class_num)
      except Exception as e:
        continue
    
  return X, Y

def loadTestingData():
  X = []
  Y = []

  for folder in FOLDERS:
    path = os.path.join(images_path, folder)
    class_num = FOLDERS.index(folder)
    min_index = TRAINING_PROPORTION*len(os.listdir(path)) *MAX_USE
    max_index = len(os.listdir(path)) *MAX_USE
    # print()
    # print(int(min_index),int(max_index))
    for img in os.listdir(path)[int(min_index):int(max_index)]:
      try:
        img_array = cv2.imread(os.path.join(path,img))
        new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        new_array = np.array(new_array)
        new_array = new_array.astype('float32')
        new_array /= 255

        X.append(new_array)
        Y.append(class_num)
      except Exception as e:
        pass
  return X, Y

# Load data

X, Y = loadTrainingData()

# XVal, YVal = loadValidationData()

XTest, YTest = loadTestingData()


# Loading Model
# model = tf.keras.load_model('/content/drive/MyDrive/HCDS Project/trained_model.h5')


# Defining Model 256 by 256 by
model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMG_SIZE,IMG_SIZE, 3), name="Input"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(2, 2), activation='relu', name="1st_Conv2D_Layer",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005)
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', name="2nd_Conv2D_Layer",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005)
                                   ),
            tf.keras.layers.Flatten(name="4th_Flatten_Layer"),
            tf.keras.layers.Dense(1, name="out", activation="sigmoid"),
        ])
# Compiling and Attaching Logger
model.compile(
              optimizer = tf.keras.optimizers.Adam(),
              loss = "binary_crossentropy",
              metrics=[
                       'accuracy',
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()
                       ]
              )

logger = tf.keras.callbacks.TensorBoard(
    log_dir='./log/' + RUN_NAME,
    write_graph=True,
    histogram_freq=1
)


# Train Model
history = model.fit(
    x=tf.cast(np.array(X), tf.float64), 
    y=tf.cast(list(map(int,Y)),tf.int32), 
    epochs=EPOCHS,
    shuffle=True,
    callbacks = [logger],
    # validation_split = 0.1,
    # validation_data = (tf.cast(np.array(XVal), tf.float64), tf.cast(list(map(int,YVal)),tf.int32),)
    # batch_size = 10,
    )


# Test Model

print("EVALUATION:")

error_rate = model.evaluate(
    x=tf.cast(np.array(XTest), tf.float64), 
    y=tf.cast(list(map(int,YTest)),tf.int32)
    )

# print(error_rate)

fileName = currdir + '/' + RUN_NAME + 'Precision=' + str(round(error_rate[2],3)) + 'Recall=' + str(round(error_rate[3],3)) + '.h5'

# fileName = currdir + '/' + RUN_NAME + '.h5'

# Saving Model
model.save(fileName)


# python3 -m tensorboard.main --logdir=/mnt/c/Users/ASUS/Desktop/NYU/2021\ Spring/Human-Centered\ Data\ Science/Final\ Project/log/ --host localhost --port 8088