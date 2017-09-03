import numpy as np
import cv2 as cv
import os

def cal_DOF(img, sigma=2):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    test = cv.GaussianBlur(img, (15, 15), sigma)
    DOF = (1.0 * img - 1.0 * test) / 255.
    return DOF


def preprocess_image(image_path, color=0, size=(288, 288)):
    """
    Read an image and convert it to a valid shape for VGG
    Return: (B, C, H, W)
    """
    img = cv.imread(image_path, color)  
    img = cv.resize(img, size)
    if len(img.shape) < 3:
        img = img[np.newaxis, np.newaxis, :, :]
    else:
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
    return img

def deprocess_image(x, size=(200, 250)):
    x = x.squeeze()
    x = np.clip(x, 0, 255).astype('uint8')
    x = cv.resize(x, size)
    return x


def get_region_mask(grid=(18, 18), layer_len=5):
    """
    Generate nose region mask
    """
    n_grid_y = 18
    n_grid_x = 18
    nose_mask = np.zeros((n_grid_y, n_grid_x))
    nose_mask[8: 14, 7: 11] = 1
    nose_mask_pool = []
    for i in reversed(range(layer_len)):
        mask_tmp = np.repeat(nose_mask, 2**i, axis=0)
        mask_tmp = np.repeat(mask_tmp, 2**i, axis=1)
        nose_mask_pool += [mask_tmp]
    return nose_mask_pool


def generate_train(photo_path, gro_path, size=(200, 250), photo2gray=False):

    inputs = []
    gros = []
    DOFs = []

    for name in sorted(os.listdir(photo_path)):
        if not name.startswith(".") and (name.endswith(".png")or name.endswith(".jpg")) and os.path.isfile(os.path.join(photo_path, name)):
            this_img = cv.imread(os.path.join(photo_path, name))
            this_gro = cv.imread(os.path.join(gro_path, name), 0)
            this_gro = cv.resize(this_gro, size)
            this_img = cv.resize(this_img, size)

            gray_img = cv.cvtColor(this_img, cv.COLOR_BGR2GRAY)
            DOF = cal_DOF(gray_img)
            img = gray_img if photo2gray else this_img
            inputs += [img]
            gros += [this_gro]
            DOFs += [DOF]

    inputs = np.array(inputs) / 255.0
    gros = np.array(gros) / 255.0
    dogs = np.array(DOFs)[..., np.newaxis]
        
    return inputs, gros, dogs


