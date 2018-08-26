# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 19:35:36 2018

@author: Baakchsu
"""

import os
from random import shuffle 
import numpy as np
from tqdm import tqdm
import cv2
TRAIN_DIRS= "G://Autonomous_Heli//crack_detection//new"
IMG_SIZE = 28

def label_img(img):
    word_label = img.split('_')[0]
   
                               
    if word_label == 'crack': return [1,0]
                                
    elif word_label == 'nocrack': return [0,1]

training_data = []
def create_train_data():
    
      i = 1
    
      for img in tqdm(os.listdir( "G://Autonomous_Heli//crack_detection//new")):
          label = label_img(img)
          path = os.path.join( "G://Autonomous_Heli//crack_detection//new",img)
          img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
          img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
          img = np.array(img).reshape(1,784)
          training_data.append([np.array(img),np.array(label)])
      shuffle(training_data)
      s = "train_data"+str(i)+".npy"
      
      np.save('G://Autonomous_Heli//crack_detection//new_npy//'+s, training_data)
    
create_train_data()

