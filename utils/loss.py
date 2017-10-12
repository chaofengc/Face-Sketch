import theano.tensor
import keras.backend as K
import numpy as np
import scipy.sparse as sp
from theano import sparse

def total_variation_loss(x, img_height=256, img_width=200):
    img_height = x.shape[1]
    img_width = x.shape[2]
    a = K.square(x[:, :img_height - 1, :img_width - 1] - x[:, 1:, :img_width - 1])
    b = K.square(x[:, :img_height - 1, :img_width - 1] - x[:, :img_height - 1, 1:])
    return K.sum(K.pow(a + b, 1.25)) / (1.*img_width*img_height)

def abs_loss(y_true, y_pred):
    e_0 = K.abs(y_pred - y_true)
    return K.mean(e_0,axis=-1) # + total_variation_loss(y_pred)

def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(S, combination,N_l):
    assert K.ndim(combination) == 3
    C = gram_matrix(combination)
    size = combination.shape[1]*combination.shape[2]
    return K.sum(K.square(S - C)) / (4. * (N_l ** 2) * (size ** 2))

def region_loss(S, combination,region_mask,N_l):
    assert K.ndim(combination) == 3
    size = K.sum(region_mask)
    combination_r = combination*region_mask
    C = gram_matrix(combination_r)
    return K.sum(K.square(S - C)) / (4. * (N_l ** 2) * (size ** 2))

def content_loss(base, combination):
    return K.sum(K.square(combination - base))


