"""
Evaluate the performance of face sketch synthesis.
Method: PCA and SVM face recognition.
Modified from face recognition examples from sklearn: 
    http://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
"""
from __future__ import print_function

from time import time
from sklearn.decomposition import PCA

import numpy as np
import cv2 as cv
import os

def load_dataset(dir_path, img_list=None, size=(200, 250)):
    data = []
    label = []
    for root, dirs, files in os.walk(dir_path):
        if img_list:
            files = img_list
        for f in sorted(files):
            img = cv.imread(os.path.join(root, f), 0)
            if img.shape != size[::-1]:
                out_size = np.array(size[::-1])
                border = out_size - img.shape
                border1 = np.floor(border / 2.).astype('int')
                border2 = np.ceil(border / 2.).astype('int')
                img = cv.copyMakeBorder(img, border1[0], border2[0], border1[1], border2[1], cv.BORDER_CONSTANT, value=255)
            data.append(img)
            label.append(f[:-4])

    return np.array(data), np.array(label)


def pred_acc(train_pca, y_train, test_pca, y_test, topk=1):
    pred = []
    train_pca = train_pca / np.linalg.norm(train_pca)
    test_pca = test_pca / np.linalg.norm(test_pca)
    
    for i in train_pca:
        dist = np.sum((test_pca - i)**2, 1)
        #  dist = 1 - np.dot(test_pca, i)
        pred.append(np.argmin(dist))
    pred = np.array(pred)
    pred_label = y_train[pred]
    return (np.sum(pred_label==y_test)*1. / y_test.shape[0])

def PCA_recognition(train_dir, test_dirs):
    X_train, y_train = load_dataset(train_dir)
    n_samples, h, w = X_train.shape
    X_train = X_train.reshape(-1, h*w)
    pca = PCA(n_components=30, svd_solver='full', whiten=True).fit(X_train)

    for test in test_dirs:
        X_test, y_test = load_dataset(test)
        X_test = X_test.reshape(-1, h*w)
        X_train_pca = pca.transform(X_train)
        X_test_pca  = pca.transform(X_test)
        print(test.split('/')[-1], '\t', pred_acc(X_train_pca, y_train, X_test_pca, y_test))


if __name__ == '__main__':
    train_dir = './Data/CUHK_student_test/sketches'
    test_dirs = ['./other_results/CUHK/MRF',
                 './other_results/CUHK/MWF',
                 './other_results/CUHK/SSD',
                 './other_results/CUHK/FCNN',
                 './other_results/CUHK/BFCN',
                 './result_CUHK/sketch']
    PCA_recognition(train_dir, test_dirs)

