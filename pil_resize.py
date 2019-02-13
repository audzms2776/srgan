import io
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps, ImageCms
from tqdm import tqdm

original_path = 'original'
resize_path = 'resize'
folder_arr = ['data']
pbar = None


def convert_to_srgb(img):
    icc = img.info.get('icc_profile', '')
    if icc:
        fd = io.BytesIO(icc)
        sPrf = ImageCms.ImageCmsProfile(fd)
        dPrf = ImageCms.createProfile('sRGB')
        img = ImageCms.profileToProfile(img, sPrf, dPrf)
    return img


def random_crop(x):
    folder = x['folder']
    image_name = x['image_name']
    img1 = Image.open(folder + '/' + image_name)

    try:
        fit_img_h = ImageOps.fit(img1, (384, 384), Image.ANTIALIAS)
        fit_img_h = convert_to_srgb(fit_img_h)
        h_name = '{}/{}_{}.png'.format(original_path, folder, image_name.split('.')[0])
        fit_img_h.save(h_name, format='PNG', icc_profile=fit_img_h.info.get('icc_profile', ''))

        fit_img_l = ImageOps.fit(img1, (96, 96), Image.ANTIALIAS)
        fit_img_l = convert_to_srgb(fit_img_l)
        l_name = '{}/{}_{}.png'.format(resize_path, folder, image_name.split('.')[0])
        fit_img_l.save(l_name, format='PNG', icc_profile=fit_img_l.info.get('icc_profile', ''))
    except:
        print('except filename: {}'.format(image_name))

    pbar.update(1)


if __name__ == "__main__":
    arr = []

    for folder in folder_arr:
        img_names = os.listdir(folder)

        for name in img_names:
            arr.append({'folder': folder, 'image_name': name})

    pbar = tqdm(total=len(arr))

    with ThreadPoolExecutor() as executor:
        executor.map(random_crop, arr)
