import time

from mona.text import lexicon, ALL_NAMES

import lzma
import pickle
import os
import cv2
import json
from PIL import Image
import numpy as np


# 把 validata 中错误的样本 -> another pickle
err_root_path = "./another_training"

def get_all_err_imgs(root_path: str):
    # 遍历文件夹下所有的图片，包括子文件夹
    all_imgs = []
    all_imgs_paths = []
    all_imgs_labels = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.abspath(os.path.join(root, file))
                img = Image.open(img_path)
                img = np.array(img)
                img_name = file.split('.')[0]
                img_label = img_name.split('_')[-1]
            
                all_imgs.append(img)
                all_imgs_labels.append(img_label)
    return all_imgs_paths, all_imgs, all_imgs_labels

paths, imgs, labels = get_all_err_imgs(err_root_path)


# 直接pickle避免多次随机读取
beg_time = time.time()
try:
    pickle.dump(paths,  lzma.open('/media/alex/Data/another_x_path.pkl', 'wb'))
    pickle.dump(imgs,   lzma.open('/media/alex/Data/another_x_imgs.pkl', 'wb'))
    pickle.dump(labels, lzma.open('/media/alex/Data/another_y.pkl', 'wb'))
except:
    pickle.dump(paths,  lzma.open('D:/another_x_path.pkl', 'wb'))
    pickle.dump(imgs,   lzma.open('D:/another_x_imgs.pkl', 'wb'))
    pickle.dump(labels, lzma.open('D:/another_y.pkl', 'wb'))
print(f'pickle dump time: {time.time() - beg_time:.4}s')
# pickle.dump(genshin_y, open('./genshin_y.pkl', 'wb'))