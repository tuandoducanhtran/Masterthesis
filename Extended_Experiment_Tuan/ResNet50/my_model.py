# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 01:00:31 2022

@author: tuanm
"""

import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D , Dense , Flatten, MaxPooling2D , Dropout , BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical 
import random
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf
import visualkeras

# image_input = Input(shape=(64, 64, 3))
# top_model = Sequential()
# top_model.add(Dense(2 , activation='softmax'))
# model = ResNet50 (input_tensor = image_input, weights = None)
# model.summary()

def My_Model(weights_path = None):
    resnet = tf.keras.applications.ResNet50(input_shape=(64, 64, 3), weights=None, include_top=False)
    X = Flatten()(resnet.output)
    # pred_gender = tf.keras.layers.Dense(2)(resnet.output)
    # pred_age = tf.keras.layers.Dense(9)(resnet.output)
    # pred_race = tf.keras.layers.Dense(5)(resnet.output)
    # pred_gender = tf.keras.layers.Dense(2)(X)
    # pred_age = tf.keras.layers.Dense(9)(X)
    # pred_race = tf.keras.layers.Dense(5)(X)
    pred_gender = tf.keras.layers.Dense(2, activation = 'softmax')(X)
    pred_age = tf.keras.layers.Dense(9, activation = 'softmax')(X)
    pred_race = tf.keras.layers.Dense(5, activation = 'softmax')(X)
    model = tf.keras.models.Model(inputs=resnet.input, outputs=[pred_gender , pred_age , pred_race])
    
    if weights_path is not None:
        model.load_weights(weights_path)


    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
				  loss=["categorical_crossentropy", "categorical_crossentropy" , "categorical_crossentropy"] ,
				  metrics=['accuracy'])
    print('Model Created') 
    print(model.summary())	
    visualkeras.layered_view(model).show()
    return model
	
if __name__=='__main__':
	My_Model(weights_path=None)