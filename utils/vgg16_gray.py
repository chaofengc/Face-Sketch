import os
import cv2 as cv
import h5py
import numpy as np
from time import time

from keras import backend as K
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.models import Sequential

class VGG16Gray(object):
    """
    Pretrained VGG16 for gray image feature extraction
    Input should be (B, C, H, W), and in [0, 255]
    """
    def __init__(self, img_size=(288, 288), weight_path='../Weight/vgg16_gray.hdf5', input_tensor=None):
        self.width = img_size[0]
        self.height = img_size[1]
        self.weight_path = weight_path
        self.pre_compile_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.build(input_tensor)

    def build(self, input_tensor=None):
        img_width = self.width
        img_height = self.height

        first_layer = ZeroPadding2D((1, 1), input_shape=(1, img_height, img_width))
        if input_tensor is not None:
            first_layer.input = input_tensor

        model = Sequential()
        model.add(first_layer)
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        # load vgg16 gray weights 
        f = h5py.File(self.weight_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()

        self.model = model
        all_layers = [l.name for l in self.model.layers]
        feature_layer_idx = [all_layers.index(fl) for fl in self.pre_compile_layers]
        feature_fun = K.function([self.model.layers[0].input], [self.model.layers[x].get_output() for x in feature_layer_idx])         
        self.feature_fun = feature_fun
       
    def get_features(self, inputs, feature_layers):
        feat_idx = [self.pre_compile_layers.index(x) for x in feature_layers]
        all_feat = self.feature_fun([inputs])
        return [all_feat[i] for i in feat_idx]

    def get_out_var(self, feature_layers):
        all_layers = [l.name for l in self.model.layers]
        idx = [all_layers.index(fl) for fl in feature_layers]
        var = [self.model.layers[x].get_output() for x in idx]
        shape = [self.model.layers[x].output_shape for x in idx]
        return var, shape

if __name__ == '__main__':
    feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    img = cv.imread('../Data/photos/74.png', 0)
    img = cv.resize(img, (288, 288))[np.newaxis, np.newaxis, ...]
    model = VGG16Gray()
    model.get_features(img, np.array(feature_layers))

