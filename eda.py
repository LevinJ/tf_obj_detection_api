#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 18:18:24 2017

@author: hyj
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from object_detection.protos import input_reader_pb2
from object_detection.builders.dataset_builder import build as build_dataset
# from utils import get_dataset


class App():
    def __init__(self):
        self.class_dist = {"classes": [],
                           "num_obj": [],
                           "box_h": [],
                           "box_w": [],
                           "box_area": []}
        return
    def get_dataset(self, tfrecord_path, label_map='label_map.pbtxt'):
       
        input_config = input_reader_pb2.InputReader()
        input_config.label_map_path = label_map
        input_config.tf_record_input_reader.input_path[:] = tfrecord_path
        input_config.num_epochs = 1
        input_config.shuffle = False
        
        dataset = build_dataset(input_config)
        return dataset
    def process_file(self, dataset):
         
        ind = 0
        for batch in dataset:
            ind = ind + 1
            print("processing id {} in {}".format(ind, batch["filename"].numpy().decode()))
            img  = batch['image'].numpy()
            h, w, _c = img.shape 
            boxes = batch['groundtruth_boxes'].numpy()
            boxes[:,(0,2)] *= h
            boxes[:,(1,3)] *= w
            box_h = boxes[:,2] - boxes[:,0]
            if (box_h > h).sum() >= 1:
                print("a bit interesting here, object is outside of image boundary")
            box_w = boxes[:,3] - boxes[:,1]
            box_area = box_h * box_w
            classes = batch['groundtruth_classes'].numpy()
            num_obj = len(classes)
            
            self.class_dist["classes"].extend(classes)
            self.class_dist["box_h"].extend(box_h)
            self.class_dist["box_w"].extend(box_w)
            self.class_dist["box_area"].extend(box_area)
            self.class_dist["num_obj"].append(num_obj)
            
        return
    def plt_charts(self):
        fig = plt.figure()
        fig.suptitle("EDA Charts", fontsize=14)

        ax = plt.subplot(2, 2, 1)
        obj_class_dist = np.array(self.class_dist["classes"])
        obj_class_dist = [(obj_class_dist == 1).sum(), (obj_class_dist == 2).sum(), (obj_class_dist == 4).sum()]
        obj_class_dist = np.array(obj_class_dist)
        obj_class_dist = obj_class_dist/float(obj_class_dist.sum())
        ax.bar(['vehicle', 'pedestrian', 'cyclist'], obj_class_dist)
        ax.set_title("object class distribution", fontsize=14)
        
        ax = plt.subplot(2, 2, 2)
        ax.hist(self.class_dist["num_obj"], density=True)
        ax.set_title("object number per image", fontsize=14)
        ax = plt.subplot(2, 2, 3)
        ax.hist(self.class_dist["box_h"], density=True)
        ax.set_title("bouding box height", fontsize=14)
        ax = plt.subplot(2, 2, 4)
        ax.hist(self.class_dist["box_w"], density=True)
        ax.set_title("bouding box width", fontsize=14)
   
        plt.show()
        return
    
  
    def run(self):
#         self.process_file('./data/train')
        tf_record_files = []
        for data_folder in ["train", "eval", "test"]:
            data_files = glob.glob('./data/{}/*.tfrecord'.format(data_folder))
            tf_record_files.extend(data_files)
        tf_record_files.sort()
        tf_dataset= self.get_dataset(tf_record_files)
        self.process_file(tf_dataset)
        self.plt_charts()
#         for file_path in training_files:
#             print("processig file {}".format(file_path))
#             self.process_file(file_path)
        
        
        
        return
if __name__ == "__main__":   
    obj= App()
    obj.run()

