#! /usr/bin/python
# -*- coding: utf8 -*-

import time

from tensorboardX import SummaryWriter

from model import *
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

    ###====================== PRE-LOAD DATA ===========================###
    train_data = (read_file_list(config.TRAIN.lr_img_path), read_file_list(config.TRAIN.hr_img_path))
    valid_data = (read_file_list(config.VALID.lr_img_path), read_file_list(config.VALID.hr_img_path))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [None, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [None, 384, 384, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True)
    net_g_test = SRGAN_g(t_image, is_train=False)
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

    try:
        saver.restore(sess, config.srgan_dir + 'model.ckpt')
        print('load checkpoint!')
    except:
        print('no checkpoint!')

    ###============================= LOAD VGG ===============================###
    vgg_target_emb.restore_params(sess)

    ##========================= initialize G ====================###

    if mode == 'g_init':
        # fixed learning rate
        sess.run(tf.assign(lr_v, lr_init))
        print(" ** fixed learning rate: %f (for init G)" % lr_init)
        for epoch in range(n_epoch_init + 1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0

            # If your machine have enough memory, please pre-load the whole train set.
            for idx in range(0, len(train_data[0]), batch_size):
                step_time = time.time()
                b_imgs_96 = tl.prepro.threading_data(train_data[0][idx:idx + batch_size], fn=read_img)
                b_imgs_384 = tl.prepro.threading_data(train_data[1][idx:idx + batch_size], fn=read_img)

                errM, _ = sess.run([mse_loss, g_optim_init],
                                   {t_image: b_imgs_96, t_target_image: b_imgs_384})

                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1

                if n_iter % 100 == 0:
                    v_imgs_96 = tl.prepro.threading_data(valid_data[0][0: batch_size], fn=read_img)
                    v_imgs_384 = tl.prepro.threading_data(valid_data[1][0: batch_size], fn=read_img)
                    out = sess.run(net_g_test.outputs, {t_image: v_imgs_96})
                    print(out.shape)
                    print("[*] save images")
                    tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)
                    tl.vis.save_images(v_imgs_384, [ni, ni], save_dir_ginit + '/true_train.png')
                    saver.save(sess, config.srgan_dir + 'model.ckpt')

            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
                epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
            print(log)
            writer.add_scalar('loss/init', total_mse_loss / n_iter, epoch)
            saver.save(sess, config.srgan_dir + 'model.ckpt')
            
        writer.close()
    else:
        ###========================= train GAN (SRGAN) =========================###
        for epoch in range(n_epoch + 1):
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
            for idx in range(0, len(train_data[0]), batch_size):
                step_time = time.time()
                b_imgs_96 = tl.prepro.threading_data(train_data[0][idx:idx + batch_size], fn=read_img)
                b_imgs_384 = tl.prepro.threading_data(train_data[1][idx:idx + batch_size], fn=read_img)

                # update D
                real_loss, fake_loss, _ = sess.run([d_loss1, d_loss2, d_optim],
                                   {t_image: b_imgs_96, t_target_image: b_imgs_384})
                ## update G
                pixel_loss, _ = sess.run([mse_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})

                print("Epoch [%2d/%2d] %4d time: %4.4fs, D(x): %.8f, D(G(x)): %.8f MSE: %.8f" %
                      (epoch, n_epoch, n_iter, time.time() - step_time, real_loss, fake_loss, pixel_loss))

                total_d_loss += real_loss
                total_g_loss += fake_loss
                n_iter += 1

                ## quick evaluation on train set
                if n_iter % 100 ==0:
                    v_imgs_96 = tl.prepro.threading_data(valid_data[0][0: batch_size], fn=read_img)
                    v_imgs_384 = tl.prepro.threading_data(valid_data[1][0: batch_size], fn=read_img)
                    out = sess.run(net_g_test.outputs, {t_image: v_imgs_96})
                    print(out.shape)
                    print("[*] save images")
                    tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
                    tl.vis.save_images(v_imgs_384, [ni, ni], save_dir_gan + '/true_train.png')
                    saver.save(sess, config.srgan_dir + 'model.ckpt')

            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
                epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                total_g_loss / n_iter)

            print(log)
            writer.add_scalar('loss/g_loss', total_d_loss / n_iter, epoch)
            writer.add_scalar('loss/d_loss', total_g_loss / n_iter, epoch)
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
