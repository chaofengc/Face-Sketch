import os
import cv2 as cv
import numpy as np

dataset = 'CUHK'
for root, dirs, files in os.walk('./other_results/{}/BFCN'.format(dataset)):
    for f in files:
        img = cv.imread(os.path.join(root, f), 0)
        border = np.array([250, 200]) - img.shape
        border1 = np.floor(border / 2.).astype('int')
        border2 = np.ceil(border / 2.).astype('int')
        img = cv.copyMakeBorder(img, border1[0], border2[0], border1[1], border2[1], cv.BORDER_CONSTANT, value=255)
        cv.imwrite(os.path.join('./other_results/{}/BFCN1'.format(dataset), f), img)
