#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahulsn
"""

from __future__ import division, print_function, absolute_import
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from . import dataparser as dp
from scipy import misc

def lstm_model(vocab_size, maxlen, shape):
    input1 = Input(shape = (shape, shape, 3))
    # Convolution layer 1
    conv1 = Conv2D(32,kernel_size = (5,5), strides = (1, 1), activation = 'relu', padding = 'same')(input1)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(norm1)
    
    
    #Convolution layer 2
    conv2 = Conv2D(64,kernel_size = (3,3), strides = (1, 1), activation = 'relu', padding = 'same')(pool1)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(norm2)
    
    
    #Convolution layer 3
    conv3 = Conv2D(64,kernel_size = (3,3), strides = (1, 1), activation = 'relu', padding = 'same')(pool2)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
    drop1 = Dropout(0.5)(pool3)
    
    
    conv4 = Conv2D(64,kernel_size = (3,3), strides = (1, 1), activation = 'relu', padding = 'same')(drop1)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)
    drop2 = Dropout(0.5)(pool4)
      
    flat1 = Flatten()(drop2)
    dense1 = Dense(256, activation='relu')(flat1)
	
    # sequence mode
    input2 = Input(shape=(maxlen,))
    
    embed1 = Embedding(vocab_size, 256, mask_zero=True)(input2)
    
    drop2 = Dropout(0.5)(embed1)
    
    lstm1 = LSTM(256)(drop2)

  #  lstm2 = LSTM(256)(lstm1)
	# decoder model
    decoder1 = add([dense1, lstm1])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[input1, input2], outputs=outputs)
	# compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
    model.summary()
    return model


# Create input Data
def preprocess_images(X_train, X_test):
    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    x_train /= 255
    x_test /= 255
#    x_train = x_train.reshape(x_train.shape[0], d, d,d, 1)
#    x_test = x_test.reshape(x_test.shape[0], d, d, d, 1)
    return (x_train, x_test)

