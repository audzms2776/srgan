import tensorflow as tf
import tensorlayer as tl
import scipy
import numpy as np
import os
from tensorlayer.prepro import *
from config import config


def read_file_list(img_dir):
    return [img_dir + x for x in os.listdir(img_dir)]

def read_tf_img(name, size):
    temp = tf.read_file(name)
    temp = tf.image.decode_image(temp, channels=3)
    temp = tf.image.per_image_standardization(temp)
    temp = tf.reshape(temp, [size, size, 3])
    
    return temp

def _parse_function(lr_name, hr_name):
    lr_img = read_tf_img(lr_name, 96)
    hr_img = read_tf_img(hr_name, 384)

    return lr_img, hr_img

def train_input_fn(lr_list, hr_list):    
    return tf.data.Dataset \
        .from_tensor_slices((lr_list, hr_list)) \
        .map(_parse_function) \
        .shuffle(buffer_size=100) \
        .batch(config.TRAIN.batch_size) \
        .make_one_shot_iterator() \
        .get_next()
        
def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def normal_img_fn(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x
