# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:35:23 2018

@author: Baakchsu
"""

import tensorflow as tf

tf.reset_default_graph()
sess=tf.Session()

import numpy as np
    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('G:/pyworks/crack_new/model.meta')
saver.restore(sess,tf.train.latest_checkpoint('G:/pyworks/crack_new'))
train=np.load('G://Autonomous_Heli//crack_detection//new_npy//train_data1.npy')
x = np.array([i[0] for i in train]).reshape(-1,784)
Y = [i[1] for i in train]
y_ = [1,0]
for i in range(1,6500):
   y_= np.vstack((y_,Y[i]))

#num = [6500,5500,4700,4600,5300,5400,3500,4400]

 

 
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
keep_prob = graph.get_tensor_by_name("prob:0")
op = graph.get_tensor_by_name("op:0")
predics = sess.run(op,feed_dict={X:x[5:6,:],keep_prob:1})
soft_out = tf.nn.softmax(predics)
print(sess.run(soft_out))
print(y_[5:6,:])