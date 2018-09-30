
import os
import numpy as np
import keras

import cv2

import donkeycar as dk
from donkeycar.parts.keras import KerasPilot


import keras
import keras.backend as K
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.engine import Layer
from tensorflow import image as tfi

class KerasStreamline(KerasPilot):
    def __init__(self, input_shape=(20, 100, 1), *args, **kwargs):
        super(KerasStreamline, self).__init__(*args, **kwargs)
        self.model = default_streamline(input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                  loss={'angle_out': 'mse', 
                        'throttle_out': 'mse'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 1.0})
        
    def run(self, img_arr):
        if img_arr is None:
            print('no image')
            return 0.0, 0.0

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        #in order to support older models with linear throttle,
        #we will test for shape of throttle to see if it's the newer
        #binned version.
        N = len(throttle[0])
        
        if N > 0:
            throttle = dk.utils.linear_unbin(throttle, N=N, offset=0.0, R=0.5)
        else:
            throttle = throttle[0][0]
        #throttle = -0.25
        angle_unbinned = dk.utils.linear_unbin(angle_binned)
        return angle_unbinned, throttle

def default_streamline(input_shape=(20, 100, 1)):
    #print ("better model")
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense    
    #import tensorflow as tf

    opt = keras.optimizers.Adam()
    drop = 0.1

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in

    x = Convolution2D(filters=8, kernel_size=(10, 10), strides=(2, 2), activation='relu')(x)
    # these kernel sizes are placeholders
    x = Convolution2D(filters=24, kernel_size=(3, 10), strides=(3, 3), activation='relu')(x)

    #x = Convolution2D(filters=8, kernel_size=(11,11), strides=(3,3), activation='relu')(x)
    #x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    #x = Convolution2D(filters=24, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    #x = Convolution2D(filters=8, kernel_size=(20,20), strides=(5,5), activation='relu')(x)
    #x = Convolution2D(filters=24, kernel_size=(16,16), strides=(1,1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='relu')(x)
    x = Dropout(rate=.1)(x)

    angle_out = Dense(units=1, activation='relu', name='angle_out')(x)
    throttle_out = Dense(units=1, activation='relu', name='throttle_out')(x)

    
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    return model



