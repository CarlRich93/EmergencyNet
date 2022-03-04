#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Input,LeakyReLU, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, UpSampling2D, Add, Concatenate, SeparableConv2D, GlobalAveragePooling2D, DepthwiseConv2D, Multiply, Reshape, Maximum, Minimum, Subtract

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras import initializers

from tensorflow.keras.models import load_model,save_model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,TerminateOnNaN,ReduceLROnPlateau, TensorBoard

from tensorflow.keras import backend as K

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet import ResNet50

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy


import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

val_data_dir='C:/Users/Carl/Documents/PhD_Year1/IndustryProject_MIND6003/EmergencyNet-master/data/AIDER/'

seed = 22
rnd.seed(seed)
np.random.seed(seed)

dsplit = 0.2

img_height=240
img_width=240
W=img_width
num_classes = 5
num_workers=1
batch_size=128
epochs = 50
lr_init=1e-1

validation_datagen = ImageDataGenerator(rescale=1./255.,
    preprocessing_function = None,validation_split=dsplit)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False, # shuffle = True leads to terrible performance!!
    )


opt = tf.keras.optimizers.SGD(lr=1e-2,momentum=0.9)
loss = CategoricalCrossentropy()

model_dir='C:/Users/Carl/Documents/PhD_Year1/IndustryProject_MIND6003/EmergencyNet-master/results/model_emergencyNet.h5'
model = load_model(model_dir)
model.load_weights(model_dir)
model.summary()
model.compile(optimizer=opt,metrics=keras.metrics.CategoricalAccuracy(),loss=loss)

score = model.evaluate(validation_generator)
print(score)


Y_pred = model.predict(validation_generator,steps =validation_generator.samples,batch_size=1)
y_pred = np.argmax(Y_pred, axis=1)


import seaborn as sns; sns.set_theme()
sns.set(font_scale=1.5)

def plot_matrix(cm, classes, title):
    ax = sns.heatmap(cm, cmap="Blues", fmt=".2f", annot=True, xticklabels=classes, yticklabels=classes, cbar=True)
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    return 0

# target_names = list(validation_generator.class_indices.keys())
target_names = ['Building', 'Fire', 'Flood', 'None', 'Traffic']
cm = confusion_matrix(validation_generator.classes, y_pred)
cm = (cm.T/sum(cm.T)).T

plot_matrix(cm,target_names,'')


print('Classification Report')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


test_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height,img_width),
    batch_size=10,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    )

x_batch, y_batch = next(test_generator)

plt.figure(figsize=(10,10))
for i in range(9):
    input_arr = img_to_array(x_batch[i])
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(x_batch[i])
    plt.title('Pred class: {classif}'.format(classif=np.argmax(predictions)))
    plt.axis("off")
