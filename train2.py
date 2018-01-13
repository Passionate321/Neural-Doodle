# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:57:00 2017

@author: PRERNA
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:33:03 2017

@author: PRERNA
"""

import dataset2

import tensorflow as tf

#import time

#from datetime import timedelta

import math

import random

import numpy as np



#Adding Seed so that random initialization is consistent

from numpy.random import seed

seed(1)      #resets seed no

from tensorflow import set_random_seed

set_random_seed(2)     

batch_size = 10

#Prepare input data

classes = ['grass','mountains','trees']

num_classes = len(classes)



# 20% of the data will automatically be used for validation

validation_size = 0.2

img_size = 128

num_channels = 3

train_path='F:\study\sem 7\major\\training_data'



# We shall load all the training and validation images and labels into memory using openCV and use that during training

data = dataset2.read_train_sets(train_path, img_size, classes, validation_size=validation_size)





print("Complete reading input data. Will Now print a snippet of it")

print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))

print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



tf.reset_default_graph()




session = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')



## labels

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1) #finding the maximum value from y_true array of 1 row







##Network graph params

filter_size_conv1 = 3 

num_filters_conv1 = 32



filter_size_conv2 = 3

num_filters_conv2 = 32



filter_size_conv3 = 3

num_filters_conv3 = 64

    

fc_layer_size = 128



def create_weights(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05)) #normally distributed variable



def create_biases(size):

    return tf.Variable(tf.constant(0.05, shape=[size])) #variable of type(constant value=.5 , shape=>no of rows)







def create_convolutional_layer(input,

               num_input_channels, 

               conv_filter_size,        

               num_filters):  

    

    ## We shall define the weights that will be trained using create_weights function.

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    ## We create biases using the create_biases function. These are also trained.

    biases = create_biases(num_filters)



    ## Creating the convolutional layer

    layer = tf.nn.conv2d(input=input,

                     filter=weights,

                     strides=[1, 1, 1, 1],

                     padding='SAME')



    layer += biases



    ## We shall be using max-pooling.  

    layer = tf.nn.max_pool(value=layer,

                            ksize=[1, 2, 2, 1],

                            strides=[1, 2, 2, 1],

                            padding='SAME')

    ## Output of pooling is fed to Relu which is the activation function for us.

    layer = tf.nn.relu(layer)



    return layer



    



def create_flatten_layer(layer):

    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] or
    #layer_shape == [num_images, img_height, img_width, num_channels]



    # But let's get it from the previous layer.

    layer_shape = layer.get_shape()



    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
   # The number of features is: img_height * img_width * num_channels

    # We can use a function from TensorFlow to calculate this.

    num_features = layer_shape[1:4].num_elements()  #Returns the number of elements in the tensor , [1:4]-> image_height,image_width,num_channels



    ## Now, we Flatten the layer so we shall have to reshape to num_features
     # Reshape the layer to [num_images, num_features].

    # Note that we just set the size of the second dimension

    # to num_features and the size of the first dimension to -1

    # which means the size in that dimension is calculated

    # so the total size of the tensor is unchanged from the reshaping.

    layer = tf.reshape(layer, [-1, num_features])



    return layer





def create_fc_layer(input,          

             num_inputs,    

             num_outputs,

             use_relu=True):

    

    #Let's define trainable weights and biases.

    weights = create_weights(shape=[num_inputs, num_outputs])

    biases = create_biases(num_outputs)



    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow

    layer = tf.matmul(input, weights) + biases

    if use_relu:

        layer = tf.nn.relu(layer)



    return layer





layer_conv1 = create_convolutional_layer(input=x,

               num_input_channels=num_channels,

               conv_filter_size=filter_size_conv1,

               num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,

               num_input_channels=num_filters_conv1,

               conv_filter_size=filter_size_conv2,

               num_filters=num_filters_conv2)



layer_conv3= create_convolutional_layer(input=layer_conv2,

               num_input_channels=num_filters_conv2,

               conv_filter_size=filter_size_conv3,

               num_filters=num_filters_conv3)

          

layer_flat = create_flatten_layer(layer_conv3)



layer_fc1 = create_fc_layer(input=layer_flat,

                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),

                     num_outputs=fc_layer_size,

                     use_relu=True)



layer_fc2 = create_fc_layer(input=layer_fc1,

                     num_inputs=fc_layer_size,

                     num_outputs=num_classes,

                     use_relu=False) 



y_pred = tf.nn.softmax(layer_fc2,name='y_pred')     #probability of each class to the output of second fully connected layer



y_pred_cls = tf.argmax(y_pred, dimension=1)         #finding the class with the max the max probability 

session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)     #to find the cost using cross entropy
                                                                                            #returns probabilty error A 1-D Tensor of length batch_size of the 
                                                                                            #same type as logits with the softmax cross
                                                                                            #entropy loss

cost = tf.reduce_mean(cross_entropy)    ##reduce_mean ->calculates mean of the array

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)   #learning rate=.0001

correct_prediction = tf.equal(y_pred_cls, y_true_cls)           #prediction is correct if predicted class and actual class matches

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #tf.cast->returns a tensor of shape of correct_prediction with dataatype float
                                                                    #accuracy ->mean of the array

session.run(tf.global_variables_initializer()) 



def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):

    acc = session.run(accuracy, feed_dict=feed_dict_train)

    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)

    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    




total_iterations = 0



saver = tf.train.Saver()

def train(num_iteration):

    global total_iterations

    

    for i in range(total_iterations,

                   total_iterations + num_iteration):


#        images labels    imagenames   class 
            
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size) # this function returns self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)



        

        feed_dict_tr = {x: x_batch,

                           y_true: y_true_batch}

        feed_dict_val = {x: x_valid_batch,

                              y_true: y_valid_batch}



        session.run(optimizer, feed_dict=feed_dict_tr)


        
        if i % int(data.train.num_examples/batch_size) == 0: 
            
            val_loss = session.run(cost, feed_dict=feed_dict_val)

            epoch = int(i / int(data.train.num_examples/batch_size))    

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)

            saver.save(session,"F:\study\sem 7\major\dogs-cats-model") 





    total_iterations += num_iteration



train(num_iteration=3000)