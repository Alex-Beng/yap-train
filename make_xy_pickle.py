import time

from mona.text import lexicon, ALL_NAMES

import lzma
import pickle
import os
import cv2
import json
from PIL import Image
import numpy as np

def js_dp(obj, path):
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

def js_ld(path):
    return json.load(open(path, 'r', encoding='utf-8'))

ALL_NAMES = ALL_NAMES + [
    # for click usage
    "进入世界申请（",
    "秘境挑战组队邀请",
]

ALL_NAMES = set(ALL_NAMES)

# 从ALL_NAMES中找到最相似的名字
def find_most_similar_name(name: str) -> str:
    if name in ALL_NAMES:
        return name
    max_sim = 0
    max_sim_name = ""
    for n in ALL_NAMES:
        sim = 0
        for c in name:
            if c in n:
                sim += 1
        if sim > max_sim:
            max_sim = sim
            max_sim_name = n
    return max_sim_name

genshin_x = js_ld('../yap/xx.json')
genshin_y = js_ld('../yap/yy.json')

# 真实标签仅使用空白数据，无需验证lexicon
def text_all_in_lexicon(text):
    if text != "" and text not in ALL_NAMES:
        return False
    for c in text:
        if c not in lexicon:
            return False
    return True

if False:
    fix_pair = []
    for i in range(len(genshin_y)):
        if not text_all_in_lexicon(genshin_y[i]):
            simi = find_most_similar_name(genshin_y[i])
            
            img = cv2.imread(os.path.join('../yap/', genshin_x[i]))
            cv2.imshow('img', img)
            print(f'{genshin_y[i]} -> {simi}')
            print(f'{genshin_x[i]} -> {genshin_y[i]}')
            k = cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if k == ord('a'):
                fix_pair.append((genshin_x[i], simi))
            elif k == ord('f'):
                fix_pair.append((genshin_x[i], fix_pair[-1][1]))
            elif k == ord('s'):
                lb = input('input label: ')
                fix_pair.append((genshin_x[i], lb))
    import json
    json.dump(fix_pair, open('fix_pair.json', 'w', encoding='utf-8'), ensure_ascii=False)
    exit()
# dumps4.0_tx/574_4_的_raw.jpg
# dumps4.0_tx/1969_4_机天正_raw.jpg
# dumps4.0_tx3/323_2_破损的面具_raw.jpg
# dumps4.0_tx4/254_4_号_raw.jpg
# dumps4.0_tx4/1110_4_的_raw.jpg
# dumps4.0_syfs/270_3_地脉的旧_raw.jpg

genshin_x = [genshin_x[i] for i in range(len(genshin_x)) if text_all_in_lexicon(genshin_y[i])]
genshin_y = [genshin_y[i] for i in range(len(genshin_y)) if text_all_in_lexicon(genshin_y[i])]

# only save the empty label xys
# genshin_x = [genshin_x[i] for i in range(len(genshin_x)) if genshin_y[i] == '']
# genshin_y = [genshin_y[i] for i in range(len(genshin_y)) if genshin_y[i] == '']

assert(len(genshin_x) == len(genshin_y))
print(f'genshin data len: {len(genshin_x)}')
root_path = "../yap/"
genshin_n = len(genshin_x)
# for speed up
# 预读入加速训练 吞吐: 500->550
genshin_y_new = []
genshin_x_path = [] # for clean up
genshin_x_imgs = []
beg_time = time.time()
for i in range(genshin_n):
    if i%100 == 0 and i != 0:
        eta_time = (time.time() - beg_time) / 100 * (genshin_n - i)
        print(f'\r{i}/{genshin_n}, eta: {eta_time:.4}s', end="")
        beg_time = time.time()
    path = os.path.join(root_path, genshin_x[i])
    with Image.open(path) as img:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        r, c = img.shape[:2]
        new_c = int(c/r*32 + 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        img = cv2.resize(img, (new_c, 32))
        # 将img中像素[min, max]->[0, 255]
        # img = img.astype(np.float32)
        # img_min = np.min(img)
        # img_max = np.max(img)
        # img = (img - img_min) / (img_max - img_min) * 255
        # img = img.astype(np.uint8)
        
        text = genshin_y[i]
        # if text == "浊水的一滴":
        #     img0 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_CONSTANT, value=0)
        #     img1 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_CONSTANT, value=255)
        #     cv2.imshow('img0', img0)
        #     k = cv2.waitKey(0)
        #     print(genshin_x[i])
        #     if k == ord('a'):
        #         for j in range(2):
        #             genshin_x_path.append(genshin_x[i])
        #             genshin_y_new.append(text)
        #             genshin_x_imgs.append(eval(f'img{j}'))
        #     else:
                
        #         continue
        if text == '':
            img0 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_CONSTANT, value=0)
            img1 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_CONSTANT, value=255)
            # img2 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_DEFAULT, value=0)
            # img3 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_REFLECT, value=0)
            # img4 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_REPLICATE, value=0)
            for j in range(2):
                genshin_x_path.append(genshin_x[i])
                genshin_y_new.append(text)
                genshin_x_imgs.append(eval(f'img{j}'))
        else:
            img0 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_CONSTANT, value=0)
            img1 = cv2.copyMakeBorder(img, 0,0,0,384-new_c, cv2.BORDER_CONSTANT, value=255)
            for j in range(2):
                genshin_x_path.append(genshin_x[i])
                genshin_y_new.append(text)
                genshin_x_imgs.append(eval(f'img{j}'))
        if path == "../yap/dumps/15165_raw.jpg":
            print("reinforce 15165_raw.jpg")
            for _ in range(10):
                for j in range(2):
                    genshin_x_path.append(genshin_x[i])
                    genshin_y_new.append(text)
                    genshin_x_imgs.append(eval(f'img{j}'))
        if path == "../yap/dumps4.0_longx3/237_2_小麦_raw.jpg":
            print("reinforce 237_2_小麦_raw.jpg")
            for _ in range(20):
                for j in range(2):
                    genshin_x_path.append(genshin_x[i])
                    genshin_y_new.append(text)
                    genshin_x_imgs.append(eval(f'img{j}'))
        


# 直接pickle避免多次随机读取
beg_time = time.time()
try:
    pickle.dump(genshin_x_path, lzma.open('/media/alex/Data/genshin_x_path.pkl', 'wb'))
    pickle.dump(genshin_x_imgs, lzma.open('/media/alex/Data/genshin_x_imgs.pkl', 'wb'))
    pickle.dump(genshin_y_new,  lzma.open('/media/alex/Data/genshin_y.pkl', 'wb'))
except:
    pickle.dump(genshin_x_path, lzma.open('D:/genshin_x_path.pkl', 'wb'))
    pickle.dump(genshin_x_imgs, lzma.open('D:/genshin_x_imgs.pkl', 'wb'))
    pickle.dump(genshin_y_new,  lzma.open('D:/genshin_y.pkl', 'wb'))
print(f'pickle dump time: {time.time() - beg_time:.4}s')
# pickle.dump(genshin_y, open('./genshin_y.pkl', 'wb'))