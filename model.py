#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorflow.keras import layers
from tensorlayer.layers import *

from utils import sn_conv, batch_norm


def SRGAN_g(t_image, is_train=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=tf.AUTO_REUSE):
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_g2(t_image, is_train=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("SRGAN_g2", reuse=tf.AUTO_REUSE):
        n = tf.layers.conv2d(t_image, 64, kernel_size=(3, 3), activation=tf.nn.relu, padding='same',
                             kernel_initializer=w_init, name='n64s1/c')

        temp = n

        # B residual blocks
        for i in range(16):
            nn = tf.layers.conv2d(n, 64, kernel_size=(3, 3), padding='same', kernel_initializer=w_init,
                                  bias_initializer=b_init,
                                  name='n64s1/c1/%s' % i)
            nn = layers.BatchNormalization(trainable=is_train, gamma_initializer=g_init,
                                           activity_regularizer=tf.nn.relu, name='n64s1/b1/%s' % i)(nn)
            nn = tf.layers.conv2d(nn, 64, kernel_size=(3, 3), padding='same', kernel_initializer=w_init,
                                  bias_initializer=b_init,
                                  name='n64s1/c2/%s' % i)
            nn = layers.BatchNormalization(trainable=is_train, gamma_initializer=g_init, name='n64s1/b2/%s' % i)(nn)
            nn = tf.add(n, nn, name='b_residual_add/%s' % i)

            n = nn

        n = tf.layers.conv2d(n, 64, kernel_size=(3, 3), padding='same', kernel_initializer=w_init,
                             bias_initializer=b_init,
                             name='n64s1/c/m')

        n = layers.BatchNormalization(trainable=is_train, gamma_initializer=g_init, name='n64s1/b/m')(n)

        n = tf.add(n, temp, name='add3')
        # B residual blacks end

        n = tf.layers.conv2d(n, 256, kernel_size=(3, 3), padding='same', kernel_initializer=w_init,
                             bias_initializer=b_init,
                             name='n256s1/1')

        # size: 96 -> 192
        n = tf.image.resize_nearest_neighbor(n, (192, 192))
        n = tf.layers.conv2d(n, 256, kernel_size=(3, 3), padding='same', kernel_initializer=w_init,
                             bias_initializer=b_init,
                             name='n256s1/2')

        # size: 192 -> 384
        n = tf.image.resize_nearest_neighbor(n, (384, 384))
        n = tf.layers.conv2d(n, 3, kernel_size=(3, 3), padding='same', kernel_initializer=w_init,
                             bias_initializer=b_init,
                             activation=tf.nn.tanh, name='out')
        return n


def SRGAN_d(input_images, is_train=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = tf.nn.leaky_relu

    with tf.variable_scope("SRGAN_d", reuse=tf.AUTO_REUSE):
        # net_in = InputLayer(input_images, name='input/images')
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
        net_h6 = sn_conv(net_h5, df_dim * 16, 1, 1, name='h6/c')
        net_h6 = batch_norm(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
        net_h7 = sn_conv(net_h6, df_dim * 8, 1, 1, name='h7/c')
        net_h7 = batch_norm(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')

        net = sn_conv(net_h7, df_dim * 2, 1, 1, name='res/c')
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
