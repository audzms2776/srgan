from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import os 
import shutil

def read_dir_imgs(dir_name):
    return [dir_name + x for x in os.listdir(dir_name)]

original_imgs = read_dir_imgs('./original/')
resize_imgs = read_dir_imgs('./resize/')

o_train, o_test, r_train, r_test = train_test_split(original_imgs, resize_imgs, test_size=0.2)

for o in o_train:
    shutil.move(o, 'train/original')

for o in o_test:
    shutil.move(o, 'valid/original')

for r in r_train:
    shutil.move(r, 'train/resize')

for r in r_test:
    shutil.move(r, 'valid/resize')