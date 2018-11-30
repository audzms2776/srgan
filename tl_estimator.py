import tensorlayer as tl

from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *


# features: lr_list
# labels: hr_list
def g_init_model(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        net_g_test = SRGAN_g(features, is_train=False)

        predictions = {
            'generated_images': net_g_test.outputs
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    net_g = SRGAN_g(features, is_train=True)

    mse_loss = tl.cost.mean_squared_error(net_g.outputs, labels, is_mean=True)
    tf.summary.scalar('g_init_mse_loss', mse_loss)
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(config.TRAIN.lr_init, trainable=False)

    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=config.TRAIN.beta1) \
        .minimize(mse_loss, var_list=g_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=mse_loss, train_op=g_optim_init)


def load_vgg(net_vgg):
    sess = tf.get_default_session()

    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)


def srgan_model(features, labels, mode, params):
    net_g = SRGAN_g(features, is_train=False)
    net_g_test = SRGAN_g(features, is_train=False)
    net_d, logits_real = SRGAN_d(labels, is_train=True)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True)

    tf.summary.image('g_init_image', net_g_test.outputs)
    tf.summary.image('label_image', labels)

    t_target_image_224 = tf.image.resize_images(
        labels, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0,
                                                 align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2)

    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, labels, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(config.TRAIN.lr_init, trainable=False)

    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=config.TRAIN.beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=config.TRAIN.beta1).minimize(d_loss, var_list=d_vars)
    joint_op = tf.group([g_optim, d_optim])

    load_vgg(net_vgg)

    return tf.estimator.EstimatorSpec(mode, loss=g_loss, train_op=joint_op)


def main(argv):
    train_lr_img_list = read_file_list(config.TRAIN.lr_img_path)
    train_hr_img_list = read_file_list(config.TRAIN.hr_img_path)

    valid_lr_img_list = read_file_list(config.VALID.lr_img_path)
    valid_hr_img_list = read_file_list(config.VALID.hr_img_path)
    ni = int(np.sqrt(config.TRAIN.batch_size)) + 1

    g_init_classifier = tf.estimator.Estimator(
        model_fn=g_init_model,
        model_dir=config.g_checkpoint_dir,
        config=tf.estimator.RunConfig(model_dir=config.g_checkpoint_dir),
        params={})

    for epoch in range(config.TRAIN.n_epoch_init):
        g_init_classifier.train(
            input_fn=lambda: train_input_fn(train_lr_img_list, train_hr_img_list))

        if (epoch + 1) % 10 == 0:
            result = g_init_classifier.predict(
                input_fn=lambda: train_input_fn(valid_lr_img_list, valid_hr_img_list))

            tl.vis.save_images(result['generated_images'], [ni, ni], config.gen_image_dir + '/train_%d.png' % epoch)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
