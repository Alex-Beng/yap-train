import os
import numpy as np
from PIL import Image
import cv2
import pickle
from random import randint, choice, uniform
from copy import deepcopy
import torch

# read all the map into memo for speed up

# 基于 __file__ 的相对路径 ../map
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MAP_CACHE_PATH = os.path.join(REPO_ROOT, 'map_cache.pkl')

_init = False
# check the Map cache exists or not
# if not, read all the map, then save it to cache
def on_init():
    global avatar_img, map_imgs
    if _init:
        return

    if os.path.exists(MAP_CACHE_PATH):
        with open(MAP_CACHE_PATH, 'rb') as f:
            map_imgs = pickle.load(f)
    else:
        MAP_PATH = os.path.join(REPO_ROOT, 'Map')
        map_imgs = []

        if os.path.exists(MAP_PATH):
            # list all *.png files
            # and read them into map_imgs
            for root, dirs, files in os.walk(MAP_PATH):
                for file in files:
                    if file.endswith('.png'):
                        img = Image.open(os.path.join(root, file))
                        # make it openCV format, with alpha channel
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
                        map_imgs.append(img)
            # save to cache
            with open(MAP_CACHE_PATH, 'wb') as f:
                pickle.dump(map_imgs, f)
        else:
            raise FileNotFoundError('Map folder not found')
    print('Map loaded:', len(map_imgs))

    # read the avatar image
    avatar_img = Image.open(os.path.join(REPO_ROOT, 'Avatar.png'))
    # convert to openCV format, with alpha channel
    avatar_img = cv2.cvtColor(np.array(avatar_img), cv2.COLOR_RGBA2BGRA)


def gen_view_mask():
    flipped = np.ones(shape=(360, 20), dtype=np.uint8)

    # cv2.imshow('flipped',flipped)

    h,w = flipped.shape

    radius = randint(70, 120)
    # print(radius)

    new_image = np.zeros(shape=(h,radius+w),dtype=np.uint8)
    h2,w2 = new_image.shape

    v_angle = randint(70, 90)
    v_beg = randint(0, 360 - v_angle)
    '''
    0->90    90->180    +90
    90->180  -180->-90  -270
    180->270 -90->0     -270
    270->360 0->90      -270
    
    '''

    # calculate the mid view angle
    mid_angle = v_beg + v_angle // 2
    # transform the angle origin point
    mid_angle = mid_angle + 90 if 0 <= mid_angle < 90 else mid_angle - 270
    
    # set the new_image
    new_image[v_beg:v_beg+v_angle, :] = 255
    # 添加从左到右的渐变
    # 添加一个阴影比例，随机增亮或变暗
    shadow_ratio = uniform(0.9, 1.1)
    # from v_beg -> v_beg+v_angle; 0 -> radius
    for i in range(v_beg, v_beg+v_angle):
        for j in range(radius):
            new_image[i, j] = 255 * (1 - j / radius)
    

    new_image[: ,w2-w:w2] = flipped
    # new_image = ~new_image
    # print(new_image.shape)

    # cv2.imshow('polar',new_image)
    # cv2.imshow('polar',new_image)

    h,w = new_image.shape

    center = (112, 112) 

    maxRadius = 112

    output= cv2.warpPolar(new_image, center=center, maxRadius=radius, dsize=(maxRadius*2,maxRadius*2), flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS )
    # print(output.shape)

    # cv2.imshow('output',output)
    # cv2.waitKey(0)

    return output, mid_angle



def generate_image(expand2polar=False, cls_head=False):
    # return 224x224 image + [angle_norm1, angle_norm2]
    # in which angle_norm in [-1, 1]
    # angle_norm = (rad - pi) / pi, rad in [0 2pi]
    '''
    algorithm:
        1. random select a map
        2. check map size > 224x224, if not, reselect until find one
        3. random crop a 224x224 image from the map
        4. random rotate the avatar
        5. "paste" the avatar to the image
        6. add weight with the view mask
    '''
    global _init
    if not _init:
        on_init()
        _init = True

    rd_map = choice(map_imgs)

    h, w = rd_map.shape[:2]
    while h < 224 or w < 224:
        rd_map = choice(map_imgs)
        h, w = rd_map.shape[:2]
        
    # random crop
    x = randint(0, w - 224)
    y = randint(0, h - 224)
    cropped_map = deepcopy(rd_map[y:y+224, x:x+224]).astype(np.float32)

    view_mask, mid_angle = gen_view_mask()
    view_mask = view_mask.astype(np.float32).reshape((224, 224, 1))

    # add weight with the view mask
    mix_ratio = uniform(0.4, 0.6)
    # mix_ratio = 0.6
    # print(cropped_map.shape, view_mask.shape)
    # cv2.imshow('crop0', cropped_map.astype(np.uint8))
    cropped_map = cropped_map * (1 - mix_ratio) + view_mask * mix_ratio
    # 还需要进行亮度补偿
    refine_ratio = 1 / mix_ratio
    # 需要按目前最大像素值约束refine_ratio
    refine_ratio = min(refine_ratio, 255 / np.max(cropped_map))
    cropped_map = cropped_map * refine_ratio
    # cv2.imshow('crop', cropped_map.astype(np.uint8))

    # random rotate the avatar image
    avt_h, avt_w = avatar_img.shape[:2]
    angle = randint(-180, 180)
    center = (avt_w // 2, avt_h // 2)
    # TODO: 加个随机的缩放？
    # -angle for clockwise
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # keep the same size
    rotated_avatar = cv2.warpAffine(avatar_img, rot_mat, (avt_w, avt_h))
    
    # just save the rgb channel
    # rotated_avatar = rotated_avatar[:, :, :3]
    alpha = rotated_avatar[:, :, 3]
    alpha_3channel = cv2.merge([alpha, alpha, alpha])
    alpha = alpha.reshape((avt_h, avt_w, 1))

    # apply the alpha channel
    rotated_avatar = rotated_avatar.astype(np.float32) * alpha / 255
    rotated_avatar = rotated_avatar.astype(np.uint8)
    
    rotated_avatar = cv2.cvtColor(rotated_avatar, cv2.COLOR_BGRA2BGR).astype(np.float32)

    # "paste" the avatar to the image, using addWeighted
    # TODO: add random roi？
    map_roi = cropped_map[112 - avt_h//2:112 + avt_h//2, 112 - avt_w//2:112 + avt_w//2].astype(np.float32)
    # add weight by alpha channel
    map_roi = map_roi * (1 - alpha / 255)
    rotated_avatar = rotated_avatar * (alpha / 255)
    map_roi = map_roi + rotated_avatar

    cropped_map[112 - avt_h//2:112 + avt_h//2, 112 - avt_w//2:112 + avt_w//2] = map_roi
    
    # print(mid_angle, angle)
    # cv2.imshow('map', cropped_map.astype(np.uint8))
    # cv2.waitKey(0)

    if expand2polar:
        # 将 224x224 展开到极坐标
        res_img = cv2.linearPolar(cropped_map, (112, 112), 112, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)
        # cv2.imshow('res_img', res_img.astype(np.uint8))
        # cv2.waitKey(0)
        
        # 对于角度，直接归一化到 [-1, 1]
        # 无需考虑数值稳定，因为已经展开到极坐标
        mid_angle, angle = mid_angle / 180, angle / 180
        label = np.array([mid_angle, angle]).astype(np.float32)
        res_img = res_img.astype(np.uint8)
        res_img = Image.fromarray(res_img)
        return res_img, label
    if cls_head:
        # 把 mid_angle 离散到 0-359
        mid_angle += 180
        mid_angle = int(mid_angle) % 360
        res_img = cropped_map.astype(np.uint8)
        res_img = Image.fromarray(res_img)
        label = torch.tensor(mid_angle)
        # print(label)
        return res_img, label

    
    # 这样norm会有数值不稳定的问题
    # mid_angle, angle = mid_angle / 180, angle / 180
    # 使用 sin, cos 代替
    mid_angles = np.sin(mid_angle / 180 * np.pi), np.cos(mid_angle / 180 * np.pi)
    angles = np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)
    # print(mid_angles, angles)
    label = np.array([*mid_angles, *angles]).astype(np.float32)
    # print(label)
    # print(cropped_map.shape)
    res_img = cropped_map.astype(np.uint8)
    # res_img = cv2.cvtColor(cropped_map.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('res_img', res_img)
    # # cv2.waitKey(0)
    res_img = Image.fromarray(res_img)
    # label = np.array([mid_angle, angle])
    return res_img, label


if __name__ == '__main__':
    while True:
        img, label = generate_image(expand2polar=True)
        # img.show()
        img = np.array(img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        print(label)