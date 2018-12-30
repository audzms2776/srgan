#! /usr/bin/python
# -*- coding: utf8 -*-

import time

import numpy as np
import tensorlayer as tl
from tensorboardX import SummaryWriter

from config import config
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
writer = SummaryWriter()

ni = int(np.sqrt(batch_size)) + 1


def train(mode):
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/ginit"
    save_dir_gan = "samples/srgan"
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [None, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [None, 384, 384, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True)
    _, logits_real = SRGAN_d(t_target_image, is_train=True)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True)

    # vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA

    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0,
                                                 align_corners=False)  # resize_generate_image_for_vgg

    vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2)
    vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2)

    # test inference
    net_g_test = SRGAN_g(t_image, is_train=False)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    # Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    # SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init_op)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_init.npz'.format(tl.global_flag['mode']),
                                 network=net_g)

    try:
        saver.restore(sess, config.srgan_dir + 'model.ckpt')
    except:
        print('no checkpoint!')

    ###============================= LOAD VGG ===============================###
    vgg_target_emb.restore_params(sess)

    ###============================= TRAINING ===============================###
    sample_imgs_96 = tl.prepro.threading_data(train_lr_imgs[0:batch_size], fn=normal_img_fn)

    ##========================= initialize G ====================###

    if mode == 'g_init':
        # fixed learning rate
        sess.run(tf.assign(lr_v, lr_init))
        print(" ** fixed learning rate: %f (for init G)" % lr_init)
        for epoch in range(0, n_epoch_init + 1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0

            # If your machine have enough memory, please pre-load the whole train set.
            for idx in range(0, len(train_hr_imgs), batch_size):
                step_time = time.time()
                b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=normal_img_fn)
                b_imgs_96 = tl.prepro.threading_data(train_lr_imgs[idx:idx + batch_size], fn=normal_img_fn)

                if isinstance(b_imgs_384, list) or isinstance(b_imgs_96, list):
                    continue

                    # update G
                errM, _ = sess.run([mse_loss, g_optim_init],
                                   {t_image: b_imgs_96, t_target_image: b_imgs_384})

                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1

            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
                epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
            print(log)
            writer.add_scalar('loss/init', total_mse_loss / n_iter, epoch)

            # quick evaluation on train set
            if (epoch != 0) and (epoch % 5 == 0):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
                print("[*] save images")
                tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_init.npz'.format(tl.global_flag['mode']),
                                  sess=sess)

        writer.close()
    else:
        ###========================= train GAN (SRGAN) =========================###
        for epoch in range(0, n_epoch + 1):
            ## update learning rate
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
                print(log)

            epoch_time = time.time()
            total_d_loss, total_g_loss, n_iter = 0, 0, 0

            # If your machine have enough memory, please pre-load the whole train set.
            for idx in range(0, len(train_hr_imgs), batch_size):
                step_time = time.time()
                b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=normal_img_fn)
                b_imgs_96 = tl.prepro.threading_data(train_lr_imgs[idx:idx + batch_size], fn=normal_img_fn)

                if isinstance(b_imgs_384, list) or isinstance(b_imgs_96, list):
                    continue

                    # update D
                errD, _ = sess.run([d_loss, d_optim],
                                   {t_image: b_imgs_96, t_target_image: b_imgs_384})
                ## update G
                errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim],
                                                     {t_image: b_imgs_96, t_target_image: b_imgs_384})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                      (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1

            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
                epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                total_g_loss / n_iter)
            print(log)
            writer.add_scalar('loss/g_loss', total_d_loss / n_iter, epoch)
            writer.add_scalar('loss/d_loss', total_g_loss / n_iter, epoch)

            ## quick evaluation on train set
            if (epoch != 0) and (epoch % 5 == 0):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
                print("[*] save images")
                tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
                saver.save(sess, config.srgan_dir + 'model.ckpt')

        writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='g_init', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'g_init':
        train('g_init')
    elif tl.global_flag['mode'] == 'srgan':
        train('srgan')
    else:
        raise Exception("Unknow --mode")
