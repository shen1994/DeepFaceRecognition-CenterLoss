# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:03:11 2018

@author: shen1994
"""

from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras import backend as K
import tensorflow as tf

from blocks import inception_block_1a
from blocks import inception_block_1b
from blocks import inception_block_1c
from blocks import inception_block_2a
from blocks import inception_block_2b
from blocks import inception_block_3a
from blocks import inception_block_3b

def inception_v2(input_shape, embedding_size=512, dropout=0.0):
    
    """
    Implementation of the Inception model used for FaceNet
    
    Arguments:
    input_shape -- shape of the images of the dataset
    Returns:
    model -- a Model() instance in Keras
    """
        
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape, name='vector_input')
    # L_input = Input((samples_classes,), name='label_input')

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # First Block
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 1, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides = 2)(X)
    
    # Second Block 64
    X = Conv2D(128, (1, 1), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)

    # Second Block 192
    X = Conv2D(192, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception 1: a/b/c
    X = inception_block_1a(X)
    X = inception_block_1b(X)
    X = inception_block_1c(X)
    
    # Inception 2: a/b
    X = inception_block_2a(X)
    X = inception_block_2b(X)
    
    # Inception 3: a/b
    X = inception_block_3a(X)
    X = inception_block_3b(X)
    
    # Top layer
    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X = Flatten()(X)
    
    # X = Dropout(dropout)(X)
    
    X = Dense(embedding_size, name='fc1')(X) 
    
    # X = tf.nn.l2_normalize(X, 1, 1e-10, name='embeddings')
    
    # X = Dense(samples_classes, name='fc2')(X)
    
    X = Lambda(lambda  x: K.l2_normalize(x, axis=-1))(X)

    # Create model instance
    model = Model(inputs = X_input, outputs = X, name='inception_v2')
        
    return model
    