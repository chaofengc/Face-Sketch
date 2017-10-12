import argparse
import os
import cv2 as cv
import numpy as np

from keras import callbacks
from keras.optimizers import *
from utils.content_model import ContentNet
from utils.img_process import generate_train
from utils.loss import abs_loss

def train(data, save_weight_dir, resume, max_epoch=300, img_size=(200, 250), batch_size=8):
    inputs, gros, dogs = data[0], data[1], data[2]

    cnet = ContentNet(img_size = img_size)
    cnet.gen_position_map(img_num = inputs.shape[0])
    inputs = np.concatenate([inputs, cnet.position_x, cnet.position_y, dogs],axis=3)

    if resume:
        cnet.model.load_weights(os.path.join(save_weight_dir, 'inception-snapshot.hdf5'))
    save_best = callbacks.ModelCheckpoint(os.path.join(save_weight_dir, 'inception-best.hdf5'), monitor='val_loss', verbose=0, save_best_only=True)
    save_snapshot = callbacks.ModelCheckpoint(os.path.join(save_weight_dir, 'inception-snapshot.hdf5'))
    opt = Adam(lr=1e-4)
    cnet.model.compile(loss=abs_loss, optimizer=opt)
    cnet.model.fit([inputs,inputs,inputs], gros, batch_size, max_epoch, validation_split=0.1, callbacks=[save_best, save_snapshot],verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training the content network')
    parser.add_argument('-f', '--facepath', type=str, default='Data/photos', help='Path for training face photos')
    parser.add_argument('-s', '--sketchpath', type=str, default='Data/sketches', help='Path for training sketch images')
    parser.add_argument('--save_weight', type=str, default='Weight/content_weight', help='Path to save content weight')
    parser.add_argument('--resume', type=int, default=0, help='resume the last training')
    parser.add_argument('--minibatch', type=int, default=8)

    arg = parser.parse_args()
    face_path = arg.facepath
    sketch_path = arg.sketchpath
    save_weight_dir = arg.save_weight
    resume = arg.resume
    batch_size = arg.minibatch

    img_size = (200, 250)

    print '===> Generating data to train'
    inputs, gros, dogs = generate_train(face_path, sketch_path, size=img_size)
    print '===> Generated data size [photo, sketch, dog]:', inputs.shape, gros.shape, dogs.shape 
    print '===> Load model and start training'
    train([inputs, gros, dogs], save_weight_dir, resume, batch_size=batch_size)

