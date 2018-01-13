# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:57:21 2017

@author: PRERNA
"""

import tensorflow as tf

import numpy as np

import os,glob,cv2

import sys,argparse





# First, pass the path of the image


image_size=128

num_channels=3

images = []

# Reading the image using OpenCV

image = cv2.imread("F:\study\sem 7\major\\testing_data\\t22.jpg") #F:\study\sem 7\major\\testing_data\\trees\\Untitled17.jpg

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resizing the image to our desired size and preprocessing will be done exactly as done during training

image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)

images.append(image)

images = np.array(images, dtype=np.uint8)

images = images.astype('float32')

images = np.multiply(images, 1.0/255.0) 

#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.

x_batch = images.reshape(1, image_size,image_size,num_channels)



## Let us restore the saved model 

sess = tf.Session()

# Step-1: Recreate the network graph. At this step only graph is created.

saver = tf.train.import_meta_graph('dogs-cats-model.meta')

# Step-2: Now let's load the weights saved using the restore method.

saver.restore(sess, tf.train.latest_checkpoint('./'))



# Accessing the default graph which we have restored

graph = tf.get_default_graph()



# Now, let's get hold of the op that we can be processed to get the output.

# In the original network y_pred is the tensor that is the prediction of the network

y_pred = graph.get_tensor_by_name("y_pred:0")   #returns the tensor with the given name



## Let's feed the images to the input placeholders

x= graph.get_tensor_by_name("x:0") 

y_true = graph.get_tensor_by_name("y_true:0") 

y_test_images = np.zeros((1, 3)) 





### Creating the feed_dict that is required to be fed to calculate y_pred 

feed_dict_testing = {x: x_batch, y_true: y_test_images}

result=sess.run(y_pred, feed_dict=feed_dict_testing)

# result is of this format [probabiliy_of_rose probability_of_sunflower]
print("[grass       mountains         trees]")
print(result)