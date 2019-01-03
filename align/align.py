# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 20:26:13 2018

@author: shen1994
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from dan_utils import get_meanshape
from dan_utils import fit_to_rect
from dan_utils import crop_resize_rotate
from dan_utils import warp_image

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    # face align model
    align_graph_def = tf.GraphDef()
    align_graph_def.ParseFromString(open("model/pico_FaceAlign_model.pb", "rb").read())
    align_tensors = tf.import_graph_def(align_graph_def, name="")
    align_sess = tf.Session()
    align_opt = align_sess.graph.get_operations()
    align_x = align_sess.graph.get_tensor_by_name("align_input:0")
    align_stage1 = align_sess.graph.get_tensor_by_name("align_stage1:0")
    align_stage2 = align_sess.graph.get_tensor_by_name("align_stage2:0")
    align_keepout1 = align_sess.graph.get_tensor_by_name("align_keepout1:0")
    align_keepout2 = align_sess.graph.get_tensor_by_name("align_keepout2:0")
    align_landmark = align_sess.graph.get_tensor_by_name("Stage2/landmark_1:0") 

    for file_name in os.listdir('../images/test_crop'):
        f_file_name = '../images/test_crop' + os.sep + file_name
        if not os.path.exists('../images/test_align/'+file_name):
            os.mkdir('../images/test_align/'+file_name)
        for image_name in os.listdir(f_file_name):
            f_image_name = f_file_name + os.sep + image_name
            rect_image = cv2.imread(f_image_name)

            align_image = np.mean(rect_image, axis=2)
            meanshape = get_meanshape()
            min_width = np.min(np.array([rect_image.shape[0], rect_image.shape[1]]))
            landmark_value = fit_to_rect(meanshape, [0, 0, align_image.shape[0]-1, align_image.shape[1]-1])
            align_image, transform = crop_resize_rotate(align_image, 112, landmark_value, meanshape)
            landmark = align_sess.run(align_landmark, feed_dict={align_x:[np.resize(align_image, (112, 112, 1))], 
                                      align_stage1:False, align_stage2:False, 
                                      align_keepout1:0.0, align_keepout2:0.0})[0]
            landmark = np.resize(landmark, (68, 2))
            landmark = np.dot(landmark - transform[1], np.linalg.inv(transform[0]))
            d_landmark = np.array([[min_width/112.0, 0], [0, min_width/112.0]])
            theta, o_image = warp_image(rect_image, landmark, np.dot(meanshape, d_landmark))
                    
            # if np.fabs(theta) <= 16.0:
            cv2.imwrite('../images/test_align/'+file_name+os.sep+image_name, o_image)
            print('../images/test_align/'+file_name+os.sep+image_name)

            
            
            
            
            
            
            
            
            
            