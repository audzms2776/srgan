import tensorflow as tf 
import numpy as np
import os
import cv2
import argparse

def read_img(name):
    x = cv2.imread(name, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

def dir_images(path):
    img_arr = [path + x for x in os.listdir(path)]
    return img_arr

parser = argparse.ArgumentParser()
parser.add_argument("mode")
args = parser.parse_args()
mode = args.mode

tf.enable_eager_execution()

true_arr = dir_images('./test/truth/')
sn_arr = dir_images('./test/test/')
sr_arr = dir_images('./test/srgan/')
bi_arr = dir_images('./test/bicubic/')

print(len(true_arr), len(sn_arr), len(sr_arr), len(bi_arr))
sr_sim_arr = []
sn_sim_arr = []
bi_sim_arr = []

for t, bi, sr, sn in zip(true_arr, bi_arr, sr_arr, sn_arr):
    t_img  = read_img(t)
    bi_img = read_img(bi)
    sr_img = read_img(sr)
    sn_img = read_img(sn)

    if mode == 'psnr':
        bi_sim = tf.image.psnr(bi_img, t_img, max_val=255).numpy()
        sr_sim = tf.image.psnr(sr_img, t_img, max_val=255).numpy()
        sn_sim = tf.image.psnr(sn_img, t_img, max_val=255).numpy()
    else:
        t_img = tf.convert_to_tensor(t_img)

        bi_sim = tf.image.ssim(tf.convert_to_tensor(bi_img), t_img, max_val=255).numpy()
        sr_sim = tf.image.ssim(tf.convert_to_tensor(sr_img), t_img, max_val=255).numpy()
        sn_sim = tf.image.ssim(tf.convert_to_tensor(sn_img), t_img, max_val=255).numpy()
    
    bi_sim_arr.append(bi_sim)
    sr_sim_arr.append(sr_sim)
    sn_sim_arr.append(sn_sim)

print('bi: ', sum(bi_sim_arr) / len(bi_sim_arr))
print('sr: ', sum(sr_sim_arr) / len(sr_sim_arr))
print('sn: ', sum(sn_sim_arr) / len(sn_sim_arr))