import numpy as np
seed = 1337  # for reproducibility
import dlib
import cv2 as cv

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, \
    Reshape, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.layers.core import ActivityRegularization
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from img_process import cal_DOF 
from loss import  abs_loss
from keras import callbacks

from theano.tensor.signal import downsample
import theano.tensor.signal as S

class ContentNet(object):
    """
    Content net used to generate content image
    Param: 
        nb_filter, default 32, filter number used in inception model
        img_size, (width, height)
    Methods:
        build(), build and compile the network
        gen_position_map(img_num), generate the position map of image
        predict(img_path, weight_path), predict the content image given photo path
    """
    def __init__(self, nb_filter=32, img_size=(200, 250)):
        self.nb_filter = nb_filter
        self.width = img_size[0]
        self.height = img_size[1]
        self.build()

    def build(self):
        inception_1 = Sequential()
        inception_2 = Sequential()

        model_1 = Sequential()
        model_3 = Sequential()
        model_5 = Sequential()

        model_1_ = Sequential()
        model_3_ = Sequential()
        model_5_ = Sequential()

        model_1.add(Convolution2D(self.nb_filter,1,1,dim_ordering='tf',activation='relu',border_mode='same',input_shape=(self.height, self.width,6)))
        model_5.add(Convolution2D(self.nb_filter,5,5,dim_ordering='tf',activation='relu',border_mode='same',input_shape=(self.height, self.width,6)))
        model_3.add(Convolution2D(self.nb_filter,3,3,dim_ordering='tf',activation='relu',border_mode='same',input_shape=(self.height, self.width,6)))
        inception_1 = Merge([model_1,model_3,model_5],mode='concat')

        inception_2_input_shape = (inception_1.output_shape[1],inception_1.output_shape[2],inception_1.output_shape[3])

        test1 = Convolution2D(self.nb_filter,1,1,dim_ordering='tf',activation='relu',border_mode='same',input_shape=inception_2_input_shape)
        test2 = Convolution2D(self.nb_filter,3,3,dim_ordering='tf',activation='relu',border_mode='same',input_shape=inception_2_input_shape)
        test3 = Convolution2D(self.nb_filter,5,5,dim_ordering='tf',activation='relu',border_mode='same',input_shape=inception_2_input_shape)

        test1.input = inception_1.get_output()
        test2.input = inception_1.get_output()
        test3.input = inception_1.get_output()

        inception_2 = Merge([test1,test2,test3],mode='concat')

        model = Sequential([inception_1])
        model.add(inception_2)
        model.add(BatchNormalization(axis=-1,gamma_init='glorot_normal'))
        model.add(Convolution2D(128,1,1,dim_ordering='tf',activation='relu',border_mode='same'))
        model.add(Convolution2D(128,1,1,dim_ordering='tf',activation='relu',border_mode='same'))
        model.add(Convolution2D(256,1,1,dim_ordering='tf',activation='relu',border_mode='same'))
        model.add(BatchNormalization(axis=-1,gamma_init='glorot_normal'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(1,3,3,dim_ordering='tf',activation='linear',border_mode='same'))
        model.add(Convolution2D(1,3,3,dim_ordering='tf',activation='linear',border_mode='same'))
        model.add(Reshape(dims=(self.height, self.width)))

        self.model = model
        self.result_func = K.function(self.model.layers[0].input, self.model.layers[-1].get_output())         

    def gen_position_map(self, img_num=1):
        position_x = range(self.width)
        position_x = np.asarray(position_x)
        position_x = np.reshape(position_x,(1,self.width))
        position_x = np.repeat(position_x,self.height,0)
        position_x = np.reshape(position_x,(1,self.height,self.width))
        position_x = np.repeat(position_x,img_num,0)
        position_x = position_x/ (1. * self.width)

        position_y = range(self.height)
        position_y = np.asarray(position_y)
        position_y = np.reshape(position_y,(self.height,1))
        position_y = np.repeat(position_y,self.width,1)
        position_y = np.reshape(position_y,(1,self.height,self.width))
        position_y = np.repeat(position_y,img_num,0)
        position_y = position_y/ (1. * self.height)

        position_x = np.expand_dims(position_x,-1)
        position_y = np.expand_dims(position_y,-1)
        self.position_x = position_x
        self.position_y = position_y

    def predict(self, img_path, weight_path):
        """
        Predict the content image of given face photo
        Params:
            img_path, path to face photo
            weight_path, path to model weight
        """
        self.gen_position_map()
        img = cv.imread(img_path)
        img = cv.resize(img, (self.width, self.height)) 
        dog = cal_DOF(img)
        img = img[np.newaxis, ...] / 255.0
        dog = dog[np.newaxis, :, :, np.newaxis]
        self.model.load_weights(weight_path)
        inputs = np.concatenate([img,self.position_x,self.position_y, dog],axis=3)
        results = self.result_func([inputs, inputs, inputs])
        return np.array(results)

if __name__== '__main__':
    img_size = (200, 250)
    content_net = ContentNet(img_size=img_size)
    #  img_dir_path = '/home/cfchen/Dropbox/face_sketch/Deepsketch/Data/test/1.png'
    #  weight_path = '/home/cfchen/Dropbox/face_sketch/Deepsketch/content_gen/inception.model'
    #  result = content_net.predict(img_path, weight_path)
    #  result = result.squeeze()
    #  cv.imshow('test', result)
    #  cv.waitKey()
    import os
    img_dir_path = '/home/cfchen/face_sketch/test'
    weight_path = '../Weight/inception.model'
    save_dir = '/home/cfchen/fast-neural-style/test'
    for root, dirs, files in os.walk(img_dir_path, topdown=False):
        for name in files:
            img_path = os.path.join(root, name)
            save_path = os.path.join(save_dir, 'content_' + name)
            result = content_net.predict(img_path, weight_path)
            result = result.squeeze() * 255
            cv.imwrite(save_path, result.astype('uint8'))
            cv.imwrite(os.path.join(save_dir, name), cv.imread(img_path))
            print save_path, 'saved'

