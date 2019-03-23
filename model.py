#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorflow.keras import layers
from tensorlayer.layers import *

from utils import sn_conv, batch_norm, parametric_relu

def upconv2d(n, size, channel, is_train):
    n = tf.image.resize_nearest_neighbor(n.outputs, size)
    n = InputLayer(n)
    n = Conv2d(n, channel, (3, 3), (1, 1), act=None, padding='SAME')
    n = BatchNormLayer(n, is_train=is_train, act=tf.nn.leaky_relu, name='norm/{}'.format(size[0]))

    return n


def SRGAN_g(t_image, is_train=False):
    with tf.variable_scope("SRGAN_g", reuse=tf.AUTO_REUSE):
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', name='n64s1/c')
        temp = n

        # residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, act=tf.nn.leaky_relu, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn
    
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, act=None, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # residual blacks end

        n = upconv2d(n, (192, 192), 32, is_train)
        n = upconv2d(n, (384, 384), 16, is_train)

        n = Conv2d(n, 3, (9, 9), (1, 1), padding='SAME')
        n = BatchNormLayer(n, is_train=is_train, act=tf.nn.tanh, name='norm_output')

        return n


def SRGAN_d(input_images, is_train=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = tf.nn.leaky_relu

    with tf.variable_scope("SRGAN_d", reuse=tf.AUTO_REUSE):
        net_h0 = sn_conv(input_images, df_dim, 4, 2, act=lrelu, name='h0/c')
        net_h1 = sn_conv(net_h0, df_dim * 2, 4, 2, name='h1/c')
        net_h1 = batch_norm(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        net_h2 = sn_conv(net_h1, df_dim * 4, 4, 2, name='h2/c')
        net_h2 = batch_norm(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
        net_h3 = sn_conv(net_h2, df_dim * 8, 4, 2, name='h3/c')
        net_h3 = batch_norm(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
        net_h4 = sn_conv(net_h3, df_dim * 16, 4, 2, name='h4/c')
        net_h4 = batch_norm(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
        net_h5 = sn_conv(net_h4, df_dim * 32, 4, 2, name='h5/c')
        net_h5 = batch_norm(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
        net_h6 = sn_conv(net_h5, df_dim * 16, 3, 1, name='h6/c')
        net_h6 = batch_norm(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
        net_h7 = sn_conv(net_h6, df_dim * 8, 3, 1, name='h7/c')
        net_h7 = batch_norm(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')

        net = sn_conv(net_h7, df_dim * 2, 3, 1, name='res/c')
        net = batch_norm(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
        net = sn_conv(net, df_dim * 2, 3, 1, name='res/c2')
        net = batch_norm(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
        net = sn_conv(net, df_dim * 8, 3, 1, name='res/c3')
        net = batch_norm(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')
        net_h8 = tf.add(net_h7, net, name='res/add')
        net_h8 = tf.nn.leaky_relu(net_h8)

        net_ho = tf.layers.flatten(net_h8, name='ho/flatten')
        net_ho = tf.layers.dense(net_ho, 1, activation=tf.identity, kernel_initializer=w_init, name='ho/dense')
        logits = net_ho
        net_ho = tf.nn.sigmoid(net_ho)

    return net_ho, logits


def Vgg19_simple_api(input_img):
    with tf.variable_scope("VGG19", reuse=tf.AUTO_REUSE):
        conv = tl.models.VGG16(input_img, end_with='pool4')

    return conv
