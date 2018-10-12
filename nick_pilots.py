
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
            metrics=['accuracy'])
        
    def run(self, img_arr):
        if img_arr is None:
            print('no image')
            return 0.0, 0.0
        print(img_arr.shape)
        
        if(img_arr.shape == 2): # grayscale
            img_arr = img_arr.reshape((1,) + img_arr.shape + (1,))
        else:
            img_arr = img_arr.reshape((1,) + img_arr.shape)

        out = self.model.predict(img_arr)
        throttle = out[0]
        angle = out[1]
        #in order to support older models with linear throttle,
        #we will test for shape of throttle to see if it's the newer
        #binned version.

        print("angle_unbinned: ")
        print(angle_unbinned) 
        return angle, throttle

def default_streamline(input_shape=(20, 100, 1)):
    from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
    from keras.models import Model
    import keras
 
    opt = keras.optimizers.Adam()
    drop = 0.1

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in

    x = Conv2D(8, kernel_size=(10, 10), strides=(2,2), 
        activation='relu', input_shape=input_shape)(x)
    x = Conv2D(24, kernel_size=(3, 23), strides=(1,1),
        activation='relu')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)
    angle_out = Dense(1, activation='relu', name='angle_out')(x)
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    return model


