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
genshin_y_imgs = []
for i in range(genshin_n):
    path = os.path.join(root_path, genshin_x[i])
    with Image.open(path) as img:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        genshin_y_imgs.append(img)

# 直接pickle避免多次随机读取
pickle.dump(genshin_y_imgs, open('./genshin_y_imgs.pkl', 'wb'))
pickle.dump(genshin_y, open('./genshin_y.pkl', 'wb'))