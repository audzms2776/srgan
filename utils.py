import os

import tensorflow as tf

from config import config

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)


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


def normal_img_fn(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x


def sn_conv(x, channels, kernel, stride, act=None, name=''):
    with tf.variable_scope(name):
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                            initializer=weight_init)
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, bias)

        if act is not None:
            x = act(x)

        return x


def batch_norm(x, act=None, is_train=False, gamma_init=0, name=''):
    x = tf.layers.batch_normalization(x, training=is_train, gamma_initializer=gamma_init, name=name)

    if act is not None:
        x = act(x)

    return x


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        # power iteration
        # Usually iteration = 1 will be enough
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
