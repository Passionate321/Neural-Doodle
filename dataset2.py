# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:56:35 2017

@author: PRERNA
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:31:48 2017

@author: PRERNA
"""

import cv2

import os

import glob #return pathnames matching a specified pattern

from sklearn.utils import shuffle  #returns shuffled array

import numpy as np





def load_train(train_path, image_size, classes):

    images = []

    labels = []

    img_names = []

    cls = []



    print('Going to read training images')

    for fields in classes:   #field->grass,mountain,tree

        index = classes.index(fields)  #index-> grass(0) ,moun(1).....

        print('Now going to read {} files (Index: {})'.format(fields, index))

        path = os.path.join(train_path, fields, '*g')

        files = glob.glob(path)

        for fl in files:

            image = cv2.imread(fl)

            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)

            image = np.multiply(image, 1.0 / 255.0)

            images.append(image)

            label = np.zeros(len(classes))

            label[index] = 1.0

            labels.append(label)

            flbase = os.path.basename(fl)

            img_names.append(flbase)

            cls.append(fields)

    images = np.array(images)

    labels = np.array(labels)

    img_names = np.array(img_names)

    cls = np.array(cls)



    return images, labels, img_names, cls





class DataSet(object):



  def __init__(self, images, labels, img_names, cls):

    self._num_examples = images.shape[0]



    self._images = images

    self._labels = labels

    self._img_names = img_names

    self._cls = cls

    self._epochs_done = 0

    self._index_in_epoch = 0



  @property

  def images(self):

    return self._images



  @property

  def labels(self):

    return self._labels



  @property

  def img_names(self):

    return self._img_names



  @property

  def cls(self):

    return self._cls



  @property

  def num_examples(self):

    return self._num_examples



  @property

  def epochs_done(self):

    return self._epochs_done



  def next_batch(self, batch_size):

    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch

    self._index_in_epoch += batch_size



    if self._index_in_epoch > self._num_examples:               #if all the images are seen once in the epoch

      # After each epoch we update this

      self._epochs_done += 1                        #increase the the mo of epoch

      start = 0

      self._index_in_epoch = batch_size

      assert batch_size <= self._num_examples  #assert works like if else throws assertion error if false

    end = self._index_in_epoch



    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

                        # till start-> end-1 



def read_train_sets(train_path, image_size, classes, validation_size):

  class DataSets(object):

    pass

  data_sets = DataSets()



  images, labels, img_names, cls = load_train(train_path, image_size, classes)

  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  



  if isinstance(validation_size, float):    #isinstance ->Returns a Boolean stating whether the
                                            #object is an instance or subclass of another object

    validation_size = int(validation_size * images.shape[0])    #vaidation_size*no of images

#images for validation -> (validation size + 1 , end)

  validation_images = images[:validation_size]  #[:1] #excluding the first value means all lines until line 1 excluded

  validation_labels = labels[:validation_size]

  validation_img_names = img_names[:validation_size]

  validation_cls = cls[:validation_size]


# training images from last parameter(valisation size)
  train_images = images[validation_size:]     #[1:] #excluding the last param mean 
                                              #all lines starting form line 1 included

  train_labels = labels[validation_size:]

  train_img_names = img_names[validation_size:]

  train_cls = cls[validation_size:]



  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)

  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)



  return data_sets