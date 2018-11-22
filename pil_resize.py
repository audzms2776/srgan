from PIL import Image, ImageOps, ImageCms
from random import randrange
from tqdm import tqdm
import os
import io 

original_path = 'original'
resize_path = 'resize'
folder_arr = next(os.walk('.'))[1]

def convert_to_srgb(img):
    icc = img.info.get('icc_profile', '')
    if icc:
        fd = io.BytesIO(icc)
        sPrf = ImageCms.ImageCmsProfile(fd)
        dPrf = ImageCms.createProfile('sRGB')
        img = ImageCms.profileToProfile(img, sPrf, dPrf)
    return img

def random_crop(folder, image_name):
    img1 = Image.open(folder + '/' + image_name)

    try:
        fit_img_h = ImageOps.fit(img1, (384, 384), Image.ANTIALIAS)
        fit_img_h = convert_to_srgb(fit_img_h)
        h_name = '{}/{}_{}.png'.format(original_path, folder, image_name.split('.')[0])
        fit_img_h.save(h_name, format='PNG', icc_profile=fit_img_h.info.get('icc_profile',''))
        
        fit_img_l = ImageOps.fit(img1, (96, 96), Image.ANTIALIAS)
        fit_img_l = convert_to_srgb(fit_img_l)
        l_name = '{}/{}_{}.png'.format(resize_path, folder, image_name.split('.')[0])
        fit_img_l.save(l_name, format='PNG', icc_profile=fit_img_l.info.get('icc_profile',''))
    except:
        pass

for folder in folder_arr:
    img_names = os.listdir(folder)
    
    for name in tqdm(img_names):
        random_crop(folder, name)

