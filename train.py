# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:08:13 2018

@author: shen1994
"""

import os
import keras
import tensorflow as tf
from keras import backend as K
from i_model import inception_v2
from generate import Generator

samples_classes = len(os.listdir("images/train_align"))
    
def center_loss(features, labels, norf_classes):

    alpha = 0.5
    features_len = features.get_shape()[1]
    centers = tf.get_variable('centers', [norf_classes, features_len], dtype=tf.float32, \
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    # calculate loss
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    # update centers    
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, labels, diff)
    
    return loss, centers

def softmax_center_loss(y_true, y_pred):
    
    sess = K.get_session()
    embed = sess.graph.get_tensor_by_name('fc1/BiasAdd:0')  
    labels = tf.cast(y_true, dtype=tf.int32)
    
    # norm loss
    norm_loss = tf.reduce_mean(tf.norm(tf.abs(embed) + 1e-4, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, norm_loss * 0.00)
    # cent loss
    cent_loss, _ = center_loss(embed, labels, samples_classes)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, cent_loss * 0.8)
    # cross loss
    logits = sess.graph.get_tensor_by_name('fc2/BiasAdd:0')
    logits = tf.reshape(logits, (-1, 1, logits.get_shape()[1]))
    cross_total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits, name='cross_total_loss')
    cross_loss = tf.reduce_mean(cross_total_loss, name='cross_loss')
    
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_loss] + regularization_losses, name='total_loss')
    
    return total_loss
    

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    epochs = 2000
    batch_size = 48
    embedding_size = 512
    image_shape = (192, 192, 3) # 96*96*3--->192*192*3--->224*224*3--->299*299*3?
     
    network = inception_v2(image_shape, samples_classes=samples_classes, embedding_size=embedding_size)
    network.load_weights('model/weights.70.hdf5', by_name=True)
    # opt = keras.optimizers.Adam(lr=5e-4) # 0.05->0.0005 # 0.1 -> 0.01
    # mini-dataset learning rate should be 0.001
    # large-dataset learning rate should be 0.03 ---> 0.003 ---> 0.0001
    opt = keras.optimizers.SGD(lr=3e-4, momentum=0.9, nesterov=True, decay=1e-6)
    network.compile(loss=softmax_center_loss, optimizer=opt)
    
    # forward network calculation，select hard-positive and hard-negtive
    # for those hard-positive, we take all positive pairs
    # for those hard-negtive, we take pairs randomly
    train_generate = Generator(path="images/train_align",
                               batch_size=batch_size,
                               image_shape=image_shape,
                               is_enhance=True)

    # backward network calculation，triplet calcution       
    callbacks = [keras.callbacks.ModelCheckpoint('model/weights.{epoch:02d}.hdf5',
                                                  verbose=1,
                                                  save_weights_only=True)]
    history = network.fit_generator(generator=train_generate.generate(), 
                                    epochs=epochs,
                                    steps_per_epoch=2000, # train_generate.source_len // batch_size,
                                    verbose=1,
                                    initial_epoch=0,
                                    callbacks=callbacks,
                                    workers=1)      
