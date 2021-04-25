# CNN Deepfake Detector

## Description

    It uses a simple Convolutional Neural Network structure built on Tensorflow.

## Dataset

    The dataset used for training comes from [Kaggle's Deepfake Detection Challenge](https://www.kaggle.com/robikscube/kaggle-deepfake-detection-introduction). It features fake images and real images. Also includes the model faces images handpicked by the team.

### /real_and_fake_face

    The /real_and_fake_face folder contains 2 subfolders:
* training_real : real facial images. Labeled by '0' in code
* training_fake : fake facial images. Labeled by '1' in code



## Model

Layer Number | Layer
--- | ---
1 | Input Layer
2 | Conv2D. layer, kernel = 55, stride = 2, #filters=32
3 | MaxPool2D, pool size = 22
4 | Conv2D. layer, kernel = 3, stride = 2, #filters=64
5 | Flatten Layer
6 | Sigmoid Layer

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metrics: Accuracy, Precision and Recall

# Files and Folders

## /real_and_fake_face

    Includes real and fake image data

## /model_train.py

    Create, train and save the model. Primary function is to perform 50 epochs for each image size to find the optimal epoch to maximize validation accuracy. 10% of data used for validation

## /model_train_and_evaluate.py

    Create, train, evaluate and save the model. Primary function is to create models of image sizes with their respective optimal epoch number. 20% of data is used for evaluation.

## /model_evaluate

    Load any existing model and re-evaluate on the evaluation dataset.

## /savedModels

    Location to save models. Currently has the final models used for discussion

### /savedModels/For EPOCH Selection

    Contains models of sizes 32, 64, 128, 256px (length of square) with EPOCHS of 50. Used to analyze the change in validation accuracy over EPOCHS.

    Currently, the following EPOCHS maximizes the respective validation accuracy for the corresponding inmage size:

* 32px: 12 EPOCHS
* 64px: 8 EPOCHS
* 128px: 4 EPOCHS
* 256px: 8 EPOCHS

### /savedModels/For IMG_SIZE Selection

    Contains models of sizes 32, 64, 128, 256px with their respectively selected EPOCHS

## /log

    Contains summaries and log files of the saved models

# Currently Saved Model's Performance

    Evaluated on the evaluation dataset. Performed in /model_train_and_evaluate.py

Image Size of Detector (px) | Accuracy | Precision | Recall
--- | --- | --- | ---
32 | 0.655 | 0.631 | 0.641
64 | 0.655 | 0.632 | 0.635
128 | 0.682 | 0.648 | 0.708
256 | 0.597 | 0.552 | 0.745