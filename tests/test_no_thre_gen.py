import os
import random

import cv2
from PIL import Image, ImageFont, ImageDraw

fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(96, 104)]


def load_bg_imgs():
    path = "../yap/dumps_full_mona/"
    # 获取文件夹下所有图片
    files = os.listdir(path)
    # 读取图片
    imgs = []
    for file in files:
        imgs.append(cv2.imread(path + file))
    return imgs


bg_imgs = load_bg_imgs()

text = "冒险家罗尔德的日志·绝云间·奥藏天池"

import os
import sys
sys.path.append(os.getcwd())

from mona.datagen.datagen import rand_color_1, rand_color_2
for i in range(10):
    color1 = rand_color_1()
    color2 = rand_color_2()
    img = Image.new("RGB", (2200 + random.randint(-150, 150), 120), color1)
    draw = ImageDraw.Draw(img)
    x = random.randint(10, 120)
    y = random.randint(-12, 20)
    sk_w = random.randint(0, 1)
    draw.text((x, y), text, color2, font=random.choice(fonts), stroke_width=sk_w)

    import numpy as np
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (384, 32))
    cv2.threshold(img, 0, 255, cv2.THRESH_OTSU, img)
    img = cv2.bitwise_not(img)
    img_inv = cv2.bitwise_not(img)

    bg_img = random.choice(bg_imgs)

    bg_r, bg_c, _ = bg_img.shape
    res_w, res_h = 384, 32

    x = np.random.randint(0, bg_c-res_w)
    y = np.random.randint(0, bg_r-res_h)

    res_img = bg_img[y:y+res_h, x:x+res_w].copy()
    # cv2.imshow("bg", res_img)

    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)

    res_img = cv2.bitwise_and(res_img, res_img, mask=img_inv)
    res_img = np.clip(res_img + img, 0, 200)
    # cv2.imshow("bg2", res_img)

    # 使用 unique 函数获取灰度值及其数量
    unique_vals, counts = np.unique(res_img, return_counts=True)

    # 找到数量最少的灰度值，如果有多个，随机选取一个
    min_count_idxs = np.where(counts == np.min(counts))[0]
    min_count_val = unique_vals[np.random.choice(min_count_idxs)]


    rand_img = np.full((32, 384), min_count_val, dtype=np.uint8)


    img = cv2.bitwise_and(rand_img, rand_img, mask=img)
    # cv2.imshow("mask rd color", img)
    # print(img.dtype, res_img.dtype)

    res_img = cv2.add(res_img, img)

    cv2.imshow("img", res_img)
    cv2.waitKey()
