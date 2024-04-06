import os
import random

import cv2


def load_bg_imgs():
    path = "../yap/dumps_full_mona2/"
    # 获取文件夹下所有图片
    files = os.listdir(path)
    # 读取图片
    imgs = []
    for file in files:
        imgs.append(cv2.imread(path + file))
    return imgs


bg_imgs = load_bg_imgs()


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

for bg_img in bg_imgs:

    top_left = (35, 853)
    but_righ = (1877, 1076)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (1645, 244)
    but_righ = (1767, 585)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (1845, 244)
    but_righ = (1892, 576)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (1604, 76)
    but_righ = (1890, 104)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (1418, 18)
    but_righ = (1774, 51)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (912, 19)
    but_righ = (1005, 56)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)


    top_left = (618, 19)
    but_righ = (1036, 100)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (835, 828)
    but_righ = (1109, 855)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (74, 533)
    but_righ = (345, 786)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)

    top_left = (33, 12)
    but_righ = (153, 47)

    # cover it with random color
    cv2.rectangle(bg_img, top_left, but_righ, random_color(), -1)


    # cv2.namedWindow("bg_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("bg_img", bg_img)
    # cv2.waitKey(0)

top_left = (368, 114)
but_righ = (1626, 810)

new_imgs = [bg_img[top_left[1]:but_righ[1], top_left[0]:but_righ[0]] for bg_img in bg_imgs]


# save images
for i, bg_img in enumerate(new_imgs):
    cv2.imwrite(f"../yap/dumps_full_mona3/{i}.jpg", bg_img)


