import tensorflow as tf
import numpy as np
import os
import cv2



import os


# Run B7 with IMG_SIZE=128EPOCHS=4Precision=0.648Recall=0.708

currdir = os.path.abspath(os.getcwd())


images_path = currdir + '/real_and_fake_face'


# CONSTANTS
FOLDERS = ["training_real", "training_fake"]
IMG_SIZE=32
TRAINING_PROPORTION = 0.8
MAX_USE = 1
EPOCHS = 6


# Functions to Load Data

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

XTest, YTest = loadTestingData()

# Loading Model
modelName = 'Run D2 with IMG_SIZE=32EPOCHS=12Precision=0.631Recall=0.641' + '.h5'
savedModelPath = currdir + '/' + modelName
# savedModelPath = currdir + '/Old Runs/' + modelName 
model = tf.keras.models.load_model(savedModelPath)



# Test Model

print("EVALUATION:\n")

error_rate = model.evaluate(
    x=tf.cast(np.array(XTest), tf.float64), 
    y=tf.cast(list(map(int,YTest)),tf.int32)
    )



# python3 -m tensorboard.main --logdir=/mnt/c/Users/ASUS/Desktop/NYU/2021\ Spring/Human-Centered\ Data\ Science/Final\ Project/log/ --host localhost --port 8088