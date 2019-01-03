# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:11:17 2018

@author: shen1994
"""

import os
import cv2
import numpy as np

class Generator(object):
    
    # people_per_batch > images_per_person: 偏向负样本，并且loss不会下降
    # people_per_batch < images_per_person: 偏向正样本，相当于每次从总样本抽取一小批次数据
    # images_per_person >= 40
    
    def __init__(self, path=None,
             batch_size=8,
             image_shape=(192, 192, 3),
             is_enhance=False):
        self.path = path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.is_enhance = is_enhance
        
        image_paths = []
        label_index = []
        source_len = 0
        labels_len = 0
        for one in os.listdir(self.path):
            label_path = self.path + os.sep + one
            print('labels load %d!' %labels_len)
            for one_i in os.listdir(label_path):
                image_paths.append(label_path + os.sep + one_i)
                label_index.append(labels_len)
                source_len += 1
            labels_len += 1
        self.image_paths = image_paths
        self.label_index = label_index
        self.source_len = source_len
        self.labels_len = labels_len
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])
        
    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * 0.5
        alpha += 1 - 0.5
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * 0.5 
        alpha += 1 - 0.5
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * 0.5
        alpha += 1 - 0.5
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * 0.5
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
  
    def horizontal_flip(self, img):
        rand = np.random.random()
        if  rand < 0.5:
            img = img[:, ::-1]
        return img
        
    def prepare_whiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def get_image(self, path):
        img = cv2.imread(path)
        if self.is_enhance:
            witch_one = np.random.randint(3)
            if witch_one == 0:
                img = self.saturation(img)
            elif witch_one == 1:
                img = self.brightness(img)
            else:
                img = self.contrast(img)
            img = self.lighting(img)
            img = self.horizontal_flip(img)
        img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
        x = self.prepare_whiten(img)

        return x
        
    def generate(self):
            
        while(True):
            
            indexes = np.arange(0, len(self.image_paths))
            np.random.shuffle(indexes)
            now_image_paths = [self.image_paths[one] for one in indexes]
            now_label_index = [self.label_index[one] for one in indexes]    
            
            counter = 0
            batch_counter = 0
            images_pixels = []
            images_labels = []
    
            print("\nGenerate " + str(self.source_len) + " Once Again!")
            
            for i in range(self.source_len):
                if (batch_counter + 1) * self.batch_size > self.source_len:
                    counter = 0
                    batch_counter = 0
                    images_pixels = []
                    images_labels = []
                    break
                images_pixels.append(self.get_image(now_image_paths[i]))
                images_labels.append(now_label_index[i])
                counter += 1
                if counter == self.batch_size:
                    yield np.array(images_pixels), np.array(images_labels)
                    
                    counter = 0
                    batch_counter +=  1
                    images_pixels = []
                    images_labels = [] 
    
    
    
    
    
    