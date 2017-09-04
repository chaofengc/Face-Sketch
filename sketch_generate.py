from scipy.misc import imread, imresize, imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b,minimize
from time import time
import os
import argparse
import h5py
import cv2 as cv
import theano as T
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from numpy import linalg

from utils.vgg16_gray import VGG16Gray
from utils.loss import *
from utils.img_process import *
from utils.content_model import ContentNet
from utils.evaluator import Evaluator
from style_generate import generate_target_style

def build_feat_function(style_layers, content_layers, component_weights, vgg_weight_path, img_size=(288, 288)):
    style_weight = component_weights[0]
    content_weight = component_weights[1]
    region_weight = component_weights[2]
    img_width, img_height = img_size
    # get tensor representations of our images
    base_image = K.variable(np.zeros(shape=(1,1,img_width,img_height)))
    target_image = K.placeholder((1, 1, img_width, img_height))
    
    # this will contain our generated image
    input_tensor = K.concatenate([base_image, target_image], axis=0)
    vgg_model = VGG16Gray(weight_path=vgg_weight_path, input_tensor=input_tensor)
    
    loss = K.variable(0.)
    # content loss
    content_features, _  = vgg_model.get_out_var(content_layers)
    base_image_features = content_features[0][0, :, :, :]
    target_features = content_features[0][1, :, :, :]
    loss += content_weight * content_loss(base_image_features, target_features)
    
    # style loss and region loss
    nose_mask_pool = get_region_mask()
    style_features, feat_shapes = vgg_model.get_out_var(style_layers)
    style_gram_var = [K.variable(np.zeros((x[1], x[1]))) for x in feat_shapes]
    nose_gram_var = [K.variable(np.zeros((x[1], x[1]))) for x in feat_shapes]
    for i in range(len(style_layers)):
        layer_features = style_features[i] 
        target_features = layer_features[1, :, :, :]
        shape = feat_shapes[i] 
    
        loss += (style_weight / len(style_layers)) * style_loss(style_gram_var[i], target_features, shape[1])
        loss += (region_weight / len(style_layers)) * region_loss(nose_gram_var[i], target_features, nose_mask_pool[i], shape[1])
    
    # get the gradients of loss with respect to the target_image 
    grads = K.gradients(loss, target_image)

    outputs = [loss]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)
    
    f_outputs = K.function([target_image], outputs)
    
    return f_outputs, (base_image, style_gram_var, nose_gram_var)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training the content network')
    parser.add_argument('-f', '--facepath', type=str, default='Data/photos', help='Path for training face photos')
    parser.add_argument('-s', '--sketchpath', type=str, default='Data/sketches', help='Path for training sketch images')
    parser.add_argument('--vggweight', type=str, default='./Weight/vgg16_gray.hdf5')
    parser.add_argument('--contentweight', type=str, default='./Weight/inception.model')
    parser.add_argument('--featpath', type=str, default='./Data/train_sketch_feat.npz')
    parser.add_argument('testpath', type=str)
    parser.add_argument('savecontent', type=str)
    parser.add_argument('savesketch', type=str)
    parser.add_argument('compweights', type=float, nargs=3)

    arg = parser.parse_args()
    content_weight_path = arg.contentweight 
    vgg_weight_path = arg.vggweight
    photo_path = arg.facepath
    sketch_path = arg.sketchpath 
    train_feat_path = arg.featpath 

    test_img_path = arg.testpath 
    save_content_path = arg.savecontent 
    save_sketch_path = arg.savesketch 
   
    component_weights = arg.compweights   # style weight, content weight, region weight
    img_width, img_height = 288, 288
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1','conv4_1','conv5_1']
    content_layers = ['conv1_1']

    print '======> Generating content image'
    content_net = ContentNet(img_size=(img_width, img_height))
    start = time()
    content_img = content_net.predict(test_img_path, content_weight_path)
    end  = time()
    print 'Content generation time', end - start
    content_img = content_img.squeeze() * 255
    cv.imwrite(save_content_path, deprocess_image(content_img))
    
    print '=====> Generating target style'
    if os.path.exists(train_feat_path):
        print 'Train feature data base already exist'
    else:
        save_train_feat(photo_path, sketch_path, vgg_weight_path, style_layers, save_path=train_feat_path)
    feat = np.load(train_feat_path)
    feat_base = [feat[x] for x in sorted(feat.files)]
    photo, sketch, _ = generate_train(photo_path, sketch_path, size=(img_width, img_height))
    photo = photo.transpose(0, 3, 1, 2)
    sketch = sketch[:, np.newaxis, :, :]
    start = time()
    target_gram, nose_gram = generate_target_style(photo, sketch, test_img_path, feat_base, style_layers, vgg_weight_path)
    end = time()
    print 'Target style generation time', end - start
    
    print '=====> Generating sketch'
    func, (base_image, style_gram_var, nose_gram_var) = build_feat_function(style_layers, content_layers, component_weights, vgg_weight_path)
    evaluator = Evaluator(func, (img_height, img_width))
    for i in range(len(target_gram)):
        style_gram_var[i].set_value(target_gram[i].astype('float32'))
        nose_gram_var[i].set_value(nose_gram[i].astype('float32'))
    
    x = content_img.copy()      # Initialization
    base_image.set_value(content_img[np.newaxis, np.newaxis, :, :].astype('float32'))
    start = time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=200,maxiter=10000)
    end = time()
    print 'Sketch optimization time', end - start
    print('Current loss value:', min_val)
    #  save current generated image
    img = deprocess_image(x.reshape((img_width, img_height)))
    cv.imwrite(save_sketch_path, img)
