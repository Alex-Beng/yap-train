from mona.text import lexicon

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


genshin_x = js_ld('../yap/xx.json')
genshin_y = js_ld('../yap/yy.json')

# 真实标签仅使用空白数据，无需验证lexicon
def text_all_in_lexicon(text):
    for c in text:
        if c not in lexicon:
            return False
    return True

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
genshin_y_imgs = []
for i in range(genshin_n):
    path = os.path.join(root_path, genshin_x[i])
    with Image.open(path) as img:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (145, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        text = genshin_y[i]
        if text == '':
            img0 = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=0)
            img1 = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=255)
            img2 = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_DEFAULT, value=0)
            img3 = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_REFLECT, value=0)
            img4 = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_REPLICATE, value=0)
            for i in range(5):
                genshin_y_new.append(text)
                genshin_y_imgs.append(eval(f'img{i}'))
        else:
            img0 = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=0)
            img1 = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=255)
            for i in range(2):
                genshin_y_new.append(text)
                genshin_y_imgs.append(eval(f'img{i}'))

# 直接pickle避免多次随机读取
pickle.dump(genshin_y_imgs, open('/media/alex/Data/genshin_y_imgs.pkl', 'wb'))
pickle.dump(genshin_y_new, open('/media/alex/Data/genshin_y.pkl', 'wb'))
# pickle.dump(genshin_y, open('./genshin_y.pkl', 'wb'))