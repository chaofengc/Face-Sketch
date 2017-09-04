import os
import numpy as np
import argparse
import cv2 as cv
from time import time

import keras.backend as K

from utils.img_process import generate_train, get_region_mask
from utils.vgg16_gray import VGG16Gray
from utils.compare_patch import compare_patch

def save_train_feat(photo_path, sketch_path, weight_path, feature_layers, save_path='./Data/train_sketch_feat.hdf5', img_size=(288, 288)): 
    photo, sketch, _ = generate_train(photo_path, sketch_path, img_size, photo2gray=True)    
    sketch = (sketch[:, np.newaxis, :, :] * 255.0).astype('float64')
    vgg16 = VGG16Gray(weight_path=weight_path)
    sketch_features = vgg16.get_features(sketch, feature_layers)
    np.savez(save_path, *sketch_features)

def generate_target_style(photo, sketch, test_path, base_pool, feature_layers, vgg_weight, compare_size=48, searching_range=6, img_size=(288, 288)):
    all_photo_pool, all_sketch_pool = (photo*255).astype('float64'), (sketch*255).astype('float64')
    photo_img = cv.imread(test_path)
    photo_img = cv.resize(photo_img, img_size).transpose(2, 0, 1).astype('float64')
    
    border_feat_net = VGG16Gray(img_size=(144, 144), weight_path=vgg_weight)

    max_find = 15
    min_find = 2
    total_imgs = all_photo_pool.shape[0]
    n_grid_y = 18
    n_grid_x = 18
    x_step = photo_img.shape[2]/n_grid_x
    y_step = photo_img.shape[1]/n_grid_y
    compare_shift = (compare_size - x_step)/2

    target_feats = [np.ndarray(x.shape[1:]) for x in base_pool]

    symb_patch_3 = np.ndarray(shape=(1,3,compare_size,compare_size)).astype('float32')
    conv_weights_3 = K.variable(symb_patch_3)
    candidate_3 = K.placeholder(shape=(total_imgs,3,compare_size+2*searching_range,compare_size+2*searching_range))
    conv_res = K.conv2d(candidate_3,conv_weights_3)
    f_conv_3 = K.function([candidate_3],conv_res)

    for jj in range(n_grid_y):
        for ii in range(n_grid_x):
            this_patch = photo_img[ :, 
                            max(0, jj*y_step - compare_shift): min(n_grid_y*y_step, (jj+1)*y_step + compare_shift),
                            max(0, ii*x_step - compare_shift): min(n_grid_x*x_step, (ii+1)*x_step + compare_shift)]
            this_patch = this_patch[np.newaxis, ...]
            if ii<min_find or ii>max_find or jj<min_find or jj>max_find:
                this_patch_rep = np.repeat(this_patch, total_imgs, 0)
                candidate_patch = all_photo_pool[:,:,max(0, jj*y_step - compare_shift): min(n_grid_y*y_step, (jj+1)*y_step + compare_shift),
                             max(0,ii*x_step - compare_shift): min(n_grid_x*x_step, (ii+1)*x_step + compare_shift)]

                diff = this_patch_rep - candidate_patch
                sq_diff = np.square(diff)
                sq_diff = np.reshape(sq_diff,(sq_diff.shape[0],sq_diff.size/sq_diff.shape[0]))
                sum_sq_diff = np.sum(sq_diff,1)
                match_idx = np.argmin(sum_sq_diff)
                for i in range(len(target_feats)):
                    x_step_i = x_step / 2**i
                    y_step_i = y_step / 2**i
                    target_feats[i][:, jj*y_step_i: (jj+1)*y_step_i, ii*x_step_i:(ii+1)*x_step_i] = base_pool[i][match_idx, :, 
                            jj*y_step_i:(jj+1)*y_step_i, ii*x_step_i:(ii+1)*x_step_i]
            else:
                candidate_patch = all_photo_pool[ :, :, 
                        max(0, jj*y_step - compare_shift - searching_range): min(n_grid_y*y_step, (jj+1)*y_step + compare_shift + searching_range),
                        max(0, ii*x_step - compare_shift - searching_range): min(n_grid_x*x_step, (ii+1)*x_step + compare_shift + searching_range)]

                diff_photo = compare_patch(this_patch, candidate_patch, x_step, searching_range, compare_size, f_conv_3, conv_weights_3)

                total_diff = diff_photo
                min_y = 0
                max_y = 2*searching_range+1
                min_x = 0
                max_x = 2*searching_range+1

                feat_jj = 4
                feat_ii = 4

                if jj<5:
                    feat_jj = jj
                    min_y = searching_range
                if jj>12:
                    feat_jj = jj-9
                    max_y = searching_range
                if ii<5:
                    feat_ii = ii
                    min_x = searching_range
                if ii>12:
                    feat_ii = ii-9
                    max_x = searching_range

                max_diff = np.max(total_diff)
                total_diff[:, min_y:max_y, min_x:max_x] = total_diff[:, min_y:max_y, min_x:max_x] - max_diff

                best_index = np.argmin(total_diff)
                best_index = np.unravel_index(best_index, total_diff.shape)

                best_patch = best_index[0]
                y_shift = best_index[1] - searching_range
                x_shift = best_index[2] - searching_range

                start_y = jj*y_step + y_shift-16*feat_jj
                start_x = ii*x_step + x_shift-16*feat_ii

                target_sketch = all_sketch_pool[best_patch, :, start_y:start_y+144, start_x:start_x+144]
                target_sketch = np.expand_dims(target_sketch, 1)
                border_patch_feat = border_feat_net.get_features(target_sketch, feature_layers) 
                for i in range(len(target_feats)):
                    x_step_i = x_step / 2**i
                    y_step_i = y_step / 2**i
                    target_feats[i][:, jj*y_step_i: (jj+1)*y_step_i, ii*x_step_i: (ii+1)*x_step_i] = border_patch_feat[i][
                            :, :, feat_jj*y_step_i:(feat_jj+1) * y_step_i, feat_ii*x_step_i: (feat_ii+1) * x_step_i]

    target_gram = []
    for f in target_feats:
        f = f.reshape((f.shape[0], f.shape[1]*f.shape[2]))
        f_gram = np.dot(f, f.transpose())
        target_gram += [f_gram]
    # generate nose region gram
    nose_mask_pool = get_region_mask()
    
    nose_gram = []
    for idx, f in enumerate(target_feats):
        f = f * nose_mask_pool[idx]
        f = f.reshape((f.shape[0], f.shape[1]*f.shape[2]))
        f_gram = np.dot(f, f.transpose())
        nose_gram += [f_gram]
    
    return target_gram, nose_gram


if __name__ == '__main__':
    
    photo_path = './Data/photos'
    sketch_path = './Data/sketches'
    vgg_weight_path = './Weight/vgg16_gray.hdf5'
    train_feat_path = './Data/train_sketch_feat.npz'
    test_path = './Data/test/1.png'
    feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    if os.path.exists(train_feat_path):
        print 'Train feature data base already exist'
    else:
        save_train_feat(photo_path, sketch_path, vgg_weight_path, feature_layers, save_path=train_feat_path)
    feat = np.load(train_feat_path)
    feat_base = [feat[x] for x in sorted(feat.files)]

    #  for idx, i in enumerate(feature_layers):
        #  tmp = np.load('../FM_train/sketch_feat%s.npy' % i)
        #  print tmp.shape, np.sum(tmp), np.sum(feat_base[idx])
        #  print np.linalg.norm(feat_base[idx] - tmp)
    #  exit()
    #  print [[x.shape, np.sum(x)] for x in feat_base]
    photo, sketch, _ = generate_train(photo_path, sketch_path, size=(288, 288))
    
    photo = photo.transpose(0, 3, 1, 2)
    sketch = sketch[:, np.newaxis, :, :]
    start = time()
    target_gram, nose_gram = generate_target_style(photo, sketch, test_path, feat_base, feature_layers, vgg_weight_path, compare_size=48, searching_range=6, img_size=(288, 288))
    end = time()
    print 'Target style time', end - start
    #  print [x.shape for x in target_gram]
    #  print [x.shape for x in nose_gram]
    


