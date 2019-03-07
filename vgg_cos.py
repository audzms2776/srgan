import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def read_img(name):
    img = cv2.imread(name)
    img = cv2.resize(img, (224, 224))
    
    return img

# data preprocess
def preprocess(queries):
    query_img = []
    
    for img_path in queries:
        img = read_img(img_path)
        query_img.append(img)

    query_img = np.asarray(query_img).astype('float32')
    query_img /= 255

    return query_img

def model_fn():
    model = Sequential()
    model_conv = VGG16(include_top=False, input_shape=(224, 224, 3))
    model.add(model_conv)

    return model

def read_img_arr(path):
    names = [os.path.join(path, x) for x in os.listdir(path)]
    return preprocess(names)

def predict_vgg_vec(model, img_arr):
    temp_vec = model.predict(img_arr, batch_size=1)
    temp_vec = temp_vec.reshape(temp_vec.shape[0], -1)
    temp_vec = l2_normalize(temp_vec)

    return temp_vec

def predict_process(model, path):
    imgs = read_img_arr(path)
    vec = predict_vgg_vec(model, imgs)

    return vec

model = model_fn()

srgan_path = 'test/srgan'
truth_path = 'test/truth'
snsrgan_path = 'test/test'
bicubic_path = 'test/bicubic'

truth_vecs = predict_process(model, truth_path) 
srgan_vecs = predict_process(model, srgan_path)
snsrgan_vecs = predict_process(model, snsrgan_path)
bicubic_vecs = predict_process(model, bicubic_path)

print(truth_vecs.shape)
print(srgan_vecs.shape)
print(snsrgan_vecs.shape)
print(bicubic_vecs.shape)

sr_ims_arr = []
snsr_ims_arr = []
bicubic_ims_arr = []

for truth, srgan, snsrgan, bicubic in zip(truth_vecs, srgan_vecs, snsrgan_vecs, bicubic_vecs):
    truth = truth.reshape(1, -1)
    srgan = srgan.reshape(1, -1)
    snsrgan = snsrgan.reshape(1, -1)
    bicubic = bicubic.reshape(1, -1)

    srgan_sim = cosine_similarity(srgan, truth)
    snsrgan_sim = cosine_similarity(snsrgan, truth)
    bicubic_sim = cosine_similarity(bicubic, truth)

    sr_ims_arr.append(srgan_sim)
    snsr_ims_arr.append(snsrgan_sim)
    bicubic_ims_arr.append(bicubic_sim)

print('bi: ', sum(bicubic_ims_arr) / len(bicubic_ims_arr))
print('sr: ', sum(sr_ims_arr) / len(sr_ims_arr))
print('snsr: ', sum(snsr_ims_arr) / len(snsr_ims_arr))
