# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:18:56 2018

@author: Anirudh

Loads input numpy data

Splits it into features and labels

Trains a 6 layer ConvNet and saves the model 

"""








import tensorflow as tf
tf.reset_default_graph()

import numpy as np
#from tqdm import tqdm
n_classes = 2
#import os
train=np.load('G://Autonomous_Heli//crack_detection//new_npy//train_data1.npy') #loads input data
X = np.array([i[0] for i in train]).reshape(-1,784) #Extracts features from the numpy dump and reshapes into (-1,784) flattened pixel array
Y = [i[1] for i in train]  #Extracts labels from the numpy dump
y_ = [1,0] #first label in the label array
for i in range(1,6500):
   y_= np.vstack((y_,Y[i]))  #Stacks the first label with the rest 


x = tf.placeholder('float', [None, 784],name='X')
x_ = tf.reshape(x, shape=[-1, 28, 28, 1])

y = tf.placeholder('float',name='Y')

keep_rate = 0.6
keep_prob = tf.placeholder(tf.float32,name="prob") #Placeholder for the dropout rate in the dropout layer



#define the weight matrices of all the layers in a dictionary 
weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'hid1':tf.Variable(tf.random_normal([1024, 512])),
               'hid2':tf.Variable(tf.random_normal([512, 256])),
               'out':tf.Variable(tf.random_normal([256, n_classes]))}
#define the bias vectors for all the layers in a dictionary data structure 
biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'hid1':tf.Variable(tf.random_normal([512])),
               'hid2':tf.Variable(tf.random_normal([256])),
               'out':tf.Variable(tf.random_normal([n_classes]))}



#define the computation flow (or) graph
conv1 = tf.nn.relu(tf.nn.conv2d(x_, weights['W_conv1'], strides=[1,1,1,1], padding='SAME') + biases['b_conv1'])
conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1,1,1,1], padding='SAME') + biases['b_conv2'])
conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
fc = tf.reshape(conv2,[-1,7*7*64])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
fc = tf.nn.dropout(fc, keep_prob)

fc = tf.nn.relu(tf.matmul(fc, weights['hid1'])+biases['hid1'])

fc = tf.nn.relu(tf.matmul(fc, weights['hid2'])+biases['hid2'])

output = tf.matmul(fc, weights['out'])
output = tf.add(output,biases['out'],name="op") #names the last layer to retrieve it during testing





cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y) )#defines the cost function
optimizer = tf.train.AdamOptimizer().minimize(cost) #optimiser to minimise the cost
    
hm_epochs = 19 #number of epochs
saver=tf.train.Saver() #defines the saver object
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    k=1
    for epoch in range(hm_epochs):
        epoch_loss = 0
        
            
            
        __, c = sess.run([optimizer, cost], feed_dict={x: X[:-300,:], y: y_[:-300,:],keep_prob:keep_rate})
        epoch_loss += c

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)) #

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:',accuracy.eval({x:X[-1500:,:], y:y_[-1500:,:],keep_prob:1}))#computes the accuracy 
    #'model' in the path below is the name of the model
    saver.save(sess,"G:/pyworks/crack_new/model") #saves the model