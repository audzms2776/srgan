import tensorlayer as tl
import tensorflow as tf 
from config import config, log_config
from utils import *
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api

    
# features: lr_list
# labels: hr_list
def g_init_model(features, labels, mode, params):
    net_g = SRGAN_g(features, is_train=True)
    net_g_test = SRGAN_g(features, is_train=False)
    
    tf.summary.image('g_init_image', net_g_test.outputs)
    tf.summary.image('label_image', labels)

    if mode == tf.estimator.ModeKeys.EVAL:
        mse_loss = tl.cost.mean_squared_error(net_g_test.outputs, labels, is_mean=True)
        return tf.estimator.EstimatorSpec(mode, loss=mse_loss)            

    tf.summary.image('g_init_image', net_g.outputs)
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, labels, is_mean=True)
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(config.TRAIN.lr_init, trainable=False)

    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=config.TRAIN.beta1) \
                            .minimize(mse_loss, var_list=g_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=mse_loss, train_op=g_optim_init)

def load_net(net_g, net_d, net_vgg):
    sess = tf.get_default_session() 
    # if tl.files.load_and_assign_npz(sess=sess, name=config.checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
    #     tl.files.load_and_assign_npz(sess=sess, name=config.checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    # tl.files.load_and_assign_npz(sess=sess, name=config.checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

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
    net_d, logits_real = SRGAN_d(labels, is_train=True)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True)

    t_target_image_224 = tf.image.resize_images(
        labels, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

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

    load_net(net_g, net_d, net_vgg)

    return tf.estimator.EstimatorSpec(mode, loss=g_loss, train_op=joint_op)
    

def main(argv):
    train_lr_img_list = read_file_list(config.TRAIN.lr_img_path)
    train_hr_img_list = read_file_list(config.TRAIN.hr_img_path)

    valid_lr_img_list = read_file_list(config.VALID.lr_img_path)
    valid_hr_img_list = read_file_list(config.VALID.hr_img_path)

    my_config = tf.estimator.RunConfig(
        save_summary_steps=1)

    # classifier = tf.estimator.Estimator(
    #     model_fn=g_init_model,
    #     config=my_config,
    #     params={})
    
    # classifier.train(
    #     input_fn=lambda :train_input_fn(train_lr_img_list, train_hr_img_list, 
    #         config.TRAIN.n_epoch_init))
    
    # classifier.evaluate(
    #     input_fn=lambda :train_input_fn(valid_lr_img_list, valid_hr_img_list, 1))
    
    srgan_classifier = tf.estimator.Estimator(
        model_fn=srgan_model,
        config=my_config,
        params={})

    srgan_classifier.train(
          input_fn=lambda :train_input_fn(valid_lr_img_list, valid_hr_img_list, 
            config.TRAIN.n_epoch_init)
    )
    
if __name__ == '__main__':
    
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)   
