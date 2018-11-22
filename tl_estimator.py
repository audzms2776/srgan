import tensorlayer as tl
import tensorflow as tf 
from config import config, log_config
from utils import *
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api

    
# features: lr_list
# labels: hr_list
def g_init_model(features, labels, mode, params):
    
    net_g = SRGAN_g(features, is_train=True, reuse=False)
    tf.summary.image('g_init_image', net_g.outputs)
    # net_g_test = SRGAN_g(features, is_train=False, reuse=True)

    mse_loss = tl.cost.mean_squared_error(net_g.outputs, labels, is_mean=True)
    tf.summary.scalar('mse_loss', mse_loss)
    
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(config.TRAIN.lr_init, trainable=False)

    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=config.TRAIN.beta1).minimize(mse_loss, var_list=g_vars,
                                                                                  global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=mse_loss, train_op=g_optim_init)

    
def load_vgg():
    sess = tf.get_default_session() 
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

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

def main(argv):
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))

    classifier = tf.estimator.Estimator(
        model_fn=g_init_model,
        params={})
    
    classifier.train(
        input_fn=lambda :train_input_fn(train_lr_img_list, train_hr_img_list))
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)   
