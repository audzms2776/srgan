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

net_g = SRGAN_g2(t_image, is_train=True)
net_g_test = SRGAN_g2(t_image, is_train=False)
_, logits_real = SRGAN_d(t_target_image, is_train=True)
_, logits_fake = SRGAN_d(net_g, is_train=True)

# vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA

t_target_image_224 = tf.image.resize_images(
    t_target_image, size=[224, 224], method=0,
    align_corners=False)
t_predict_image_224 = tf.image.resize_images(net_g, size=[224, 224], method=0,
                                                align_corners=False)  # resize_generate_image_for_vgg

vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2)
vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2)

# ###========================== DEFINE TRAIN OPS ==========================###
real_predict = tf.equal(logits_real, tf.ones_like(logits_real))
fake_predict = tf.equal(logits_fake, tf.zeros_like(logits_fake))

real_acc = tf.reduce_mean(tf.cast(real_predict, tf.float32))
fake_acc = tf.reduce_mean(tf.cast(fake_predict, tf.float32))

d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
d_loss = d_loss1 + d_loss2

g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
mse_loss = tl.cost.mean_squared_error(net_g, t_target_image, is_mean=True)
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
    print('load checkpoint')
except:
    print('no checkpoint!')

cnt = 0
print(len(valid_data[0]))

for idx in range(0, len(valid_data[0]), batch_size):
    step_time = time.time()
    print(idx)
    v_imgs_96 = tl.prepro.threading_data(valid_data[0][idx : idx+batch_size], fn=read_img)
    
    test_result = sess.run([net_g_test], {t_image: v_imgs_96})
    
    for test in test_result[0]:
        re_test = np.array([test])
        print(re_test.shape)
        tl.vis.save_images(re_test, [1, 1], 'test/{}.png'.format(cnt))
        cnt += 1