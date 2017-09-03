from __future__ import print_function
from scipy.misc import imread, imresize, imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b,minimize
import time
import os
import argparse
import h5py
import cv2 as cv

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from numpy import linalg

import json, io


def compare_patch(this_patch,candidate_region,step,searching_range,compare_size,f_conv,conv_weights):
    total_patch = np.sum(this_patch**2)
    cum_y_candidate = np.cumsum(candidate_region**2, axis=2)
    cum_xy_candidate = np.cumsum(cum_y_candidate, axis=3)
    cum_xy_candidate = np.sum(cum_xy_candidate,1)

    cum_xy_candidate = np.append(np.zeros([cum_xy_candidate.shape[0],1,cum_xy_candidate.shape[2]]),cum_xy_candidate,1)
    cum_xy_candidate = np.append(np.zeros([cum_xy_candidate.shape[0],cum_xy_candidate.shape[1],1]),cum_xy_candidate,2)

    sum_square_candidate = cum_xy_candidate[:,0:2*searching_range+1,0:2*searching_range+1]\
                        +cum_xy_candidate[:,compare_size:compare_size+2*searching_range+1
                        ,compare_size:compare_size+2*searching_range+1]\
                        -cum_xy_candidate[:,0:2*searching_range+1,
                         compare_size:compare_size+2*searching_range+1]\
                        -cum_xy_candidate[:,compare_size:compare_size+2*searching_range+1,
                         0:2*searching_range+1]

    this_patch = this_patch[:,:,::-1,::-1]
    conv_weights.set_value(this_patch.astype('float32'))
    cross_sum = f_conv([candidate_region])
    cross_sum = cross_sum[:,0,:,:]

    diff_photo = sum_square_candidate+total_patch-2*cross_sum
    return diff_photo
