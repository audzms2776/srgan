from easydict import EasyDict as edict
import json
import tensorlayer as tl

config = edict()
config.TRAIN = edict()

# Adam
config.TRAIN.batch_size = 15
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

# initialize G
config.TRAIN.n_epoch_init = 100
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

# adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

# train set location
config.TRAIN.hr_img_path = 'train/original/'
config.TRAIN.lr_img_path = 'train/resize/'

config.VALID = edict()
# test set location
config.VALID.hr_img_path = 'valid/original/'
config.VALID.lr_img_path = 'valid/resize/'

# tpu location
config.gs_dir = 'gs://srganimagedata/data/'

# config location
config.tpu_init_dir = 'gs://srganimagedata/tes1'
config.tpu_srgan_dir = 'gs://srganimagedata/pre'
config.gen_image_dir = './checkpoint/generate/'
config.srgan_dir = './srgan_checkpoint/'

tl.files.exists_or_mkdir(config.gen_image_dir)
tl.files.exists_or_mkdir(config.srgan_dir)

# save constant
config.NUM_VIZ_IMAGES = 30


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
