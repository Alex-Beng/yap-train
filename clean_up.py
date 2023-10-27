# 使用合成数据训练好的模型来清理数据
import random
import os
import time
from copy import deepcopy

from PIL import Image, ImageFont, ImageDraw
import numpy as np

from mona.text import lexicon, index_to_word
from mona.nn.model2 import Model2

from mona.nn import predict as predict_net

import pickle
import cv2
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import argparse
import pyperclip


def js_dp(obj, path):
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

def js_ld(path):
    return json.load(open(path, 'r', encoding='utf-8'))


genshin_x = pickle.load(open('/media/alex/Data/genshin_x_imgs.pkl', 'rb'))
genshin_x_path = pickle.load(open('/media/alex/Data/genshin_x_path.pkl', 'rb'))
genshin_y = pickle.load(open('/media/alex/Data/genshin_y.pkl', 'rb'))

# genshin_x = genshin_x[::-1]
# genshin_x_path = genshin_x_path[::-1]
# genshin_y = genshin_y[::-1]

root_path = "../yap/"
genshin_n = len(genshin_x)

parser = argparse.ArgumentParser(
        description='validate data using a model')
parser.add_argument('model_file', type=str,
                    help='The model file. e.g. model_training.pt')
args = parser.parse_args()
model_file_path = args.model_file

device = "cuda"
device = "cpu"
net = Model2(len(index_to_word), 1).to(device)
net.load_state_dict(torch.load(
        model_file_path, map_location=torch.device(device)))
net.eval()
with torch.no_grad():
    begin_time = time.time()
    i = 0
    while i < genshin_n:
        if i%100 == 0 and i != 0:
            end_time = time.time()
            tput = 100/(end_time-begin_time)
            begin_time = end_time
            #  保留两位精度
            print(f"i={i} tput={tput:.2f}", end='\r')
        # path = os.path.join(root_path, genshin_x[i])
        # if genshin_y[i] == '蕈兽孢子':
        #     cv2.imshow("c", genshin_x[i])
        #     print(genshin_x_path[i])
        #     k = cv2.waitKey(0)
        #     if k == ord('q'):
        #         exit()
        path = genshin_x_path[i]

        img = genshin_x[i]
        img_cv = deepcopy(img)
        img = Image.fromarray(img)
        im = transforms.ToTensor()(img)
        im = im.unsqueeze(0)
        im = im.to(device)
        predict = predict_net(net, im)[0]

        if predict != genshin_y[i]:
            print(f"unpair: {path} {predict==genshin_y[i]}\n =={predict}== vs =={genshin_y[i]}==")
            cv2.imshow("unpair", img_cv)
            k = cv2.waitKey(0)
            if k == ord('q'):
                exit()
            else:
                # 复制path到剪切板
                pyperclip.copy(f"\n'{path}',")
        if genshin_y[i] == '':
            i += 5
        else:
            i += 2

# wrong label
'''
unpair: ../yap/ dumps/17_raw.jpg   
unpair: ../yap/ dumps/1773_raw.jpg 
unpair: ../yap/ dumps/2878_raw.jpg 
unpair: ../yap/ dumps/3079_raw.jpg 
unpair: ../yap/ dumps/3421_raw.jpg 
unpair: ../yap/ dumps/9279_raw.jpg 
unpair: ../yap/ dumps/13062_raw.jpg
unpair: ../yap/ dumps/14761_raw.jpg

unpair: ../yap/ text_dumps/12_raw.jpg
unpair: ../yap/ text_dumps/13_raw.jpg
unpair: ../yap/ text_dumps/14_raw.jpg
unpair: ../yap/ text_dumps/15_raw.jpg
unpair: ../yap/ text_dumps/25_raw.jpg
unpair: ../yap/ text_dumps/26_raw.jpg
unpair: ../yap/ text_dumps/28_raw.jpg
unpair: ../yap/ text_dumps/29_raw.jpg
unpair: ../yap/ text_dumps/5496_raw.jpg
unpair: ../yap/ text_dumps/5497_raw.jpg
unpair: ../yap/ text_dumps/5498_raw.jpg
unpair: ../yap/ text_dumps/8798_raw.jpg
unpair: ../yap/ text_dumps/8799_raw.jpg
unpair: ../yap/ text_dumps/11261_raw.jpg
unpair: ../yap/ text_dumps/15279_raw.jpg
unpair: ../yap/ text_dumps/23792_raw.jpg
unpair: ../yap/ text_dumps/23794_raw.jpg
unpair: ../yap/ text_dumps/32526_raw.jpg

unpair: ../yap/ dumps3/363_混沌容器_raw.jpg
unpair: ../yap/ dumps3/1198_兽肉_raw.jpg
unpair: ../yap/ dumps3/1379_薄荷_raw.jpg
unpair: ../yap/ dumps3/1661_簇_raw.jpg
unpair: ../yap/ dumps3/1868_教官的怀表_raw.jpg

unpair: ../yap/./ dumps4.0/733_3_的_raw.jpg
unpair: ../yap/./ dumps4.0/1022_2_浊水的一_raw.jpg
unpair: ../yap/./ dumps4.0_tx/542_2_异海凝珠_raw.jpg
unpair: ../yap/./ dumps4.0_tx/725_2_游医的怀钟_raw.jpg
unpair: ../yap/./ dumps4.0_tx/1001_2_调查_raw.jpg
unpair: ../yap/./ dumps4.0_tx/1516_4_的时_raw.jpg

'''
'''
unpair: ../yap/dumps/14777_raw.jpg
unpair: ../yap/dumps/14789_raw.jpg
unpair: ../yap/dumps/14810_raw.jpg
unpair: ../yap/dumps/15004_raw.jpg
unpair: ../yap/dumps/15163_raw.jpg
unpair: ../yap/dumps/16059_raw.jpg
unpair: ../yap/text_dumps/8_raw.jpg
unpair: ../yap/text_dumps/9_raw.jpg
unpair: ../yap/text_dumps/10_raw.jpg
unpair: ../yap/text_dumps/11_raw.jpg
unpair: ../yap/text_dumps/27_raw.jpg
unpair: ../yap/text_dumps/4216_raw.jpg
unpair: ../yap/text_dumps/4401_raw.jpg
unpair: ../yap/text_dumps/6356_raw.jpg
unpair: ../yap/text_dumps/23725_raw.jpg
unpair: ../yap/dumps4.0/1002_2_浊水的一_raw.jpg
unpair: ../yap/dumps4.0/1007_2_浊水的一_raw.jpg
unpair: ../yap/dumps4.0/1022_2_浊水的一_raw.jpg
unpair: ../yap/dumps4.0/1027_2_浊水的一_raw.jpg
unpair: ../yap/dumps4.0/1268_3_浊水的一_raw.jpg
unpair: ../yap/dumps4.0_tx/88_2_浊水的一_raw.jpg
unpair: ../yap/dumps4.0_tx/1943_3_浊水的一_raw.jpg
'''
'''
unpair: ../yap/dumps/563_raw.jpg
unpair: ../yap/dumps/3713_raw.jpg
unpair: ../yap/dumps/6703_raw.jpg
unpair: ../yap/dumps/12515_raw.jpg
unpair: ../yap/dumps/14750_raw.jpg
unpair: ../yap/dumps/14760_raw.jpg
unpair: ../yap/dumps/14761_raw.jpg
unpair: ../yap/dumps/14777_raw.jpg
unpair: ../yap/dumps/14789_raw.jpg
unpair: ../yap/dumps/14808_raw.jpg
unpair: ../yap/dumps/14810_raw.jpg
unpair: ../yap/dumps/15135_raw.jpg
unpair: ../yap/dumps/15192_raw.jpg
unpair: ../yap/text_dumps/6_raw.jpg
unpair: ../yap/text_dumps/7_raw.jpg
unpair: ../yap/text_dumps/9153_raw.jpg
unpair: ../yap/text_dumps/25105_raw.jpg
unpair: ../yap/dumps4.0_tx/18_3_浊水的一_raw.jpg
unpair: ../yap/dumps4.0_tx/54_3_浊水的_raw.jpg
'''

'''
unpair: ../yap/dumps/16061_raw.jpg
unpair: ../yap/text_dumps/9153_raw.jpg
unpair: ../yap/dumps3/227_战脉的枯叶_raw.jpg
unpair: ../yap/dumps4.0_tx/175_2_浊水的一_raw.jpg False
unpair: ../yap/dumps4.0_tx/535_2_异海凝_raw.jpg False
'''

'''
unpair: ../yap/dumps/15006_raw.jpg False
unpair: ../yap/dumps/15018_raw.jpg False
unpair: ../yap/dumps/15111_raw.jpg False
unpair: ../yap/dumps/15198_raw.jpg False
unpair: ../yap/dumps4.0/1477_2_异海凝_raw.jpg False
unpair: ../yap/dumps4.0_tx/87_2_浊水的一_raw.jpg False
'''

'''
unpair: ../yap/dumps/3277_raw.jpg
unpair: ../yap/dumps/3676_raw.jpg
unpair: ../yap/dumps/9280_raw.jpg
unpair: ../yap/dumps/13528_raw.jpg
unpair: ../yap/dumps/14751_raw.jpg
unpair: ../yap/dumps/15050_raw.jpg
unpair: ../yap/dumps/15165_raw.jpg
unpair: ../yap/dumps/16160_raw.jpg
unpair: ../yap/dumps/18382_raw.jpg
unpair: ../yap/dumps/18384_raw.jpg
unpair: ../yap/dumps/18401_raw.jpg
阅读「圣章石」标签
圣章石标签的
e:\Dev\yap\dumps\18401_raw.jpg 
e:\Dev\yap\dumps\18409_raw.jpg 
e:\Dev\yap\dumps\18408_raw.jpg
e:\Dev\yap\dumps\18407_raw.jpg 
e:\Dev\yap\dumps\18406_raw.jpg 
e:\Dev\yap\dumps\18405_raw.jpg 
e:\Dev\yap\dumps\18404_raw.jpg 
e:\Dev\yap\dumps\18403_raw.jpg 
e:\Dev\yap\dumps\18402_raw.jpg
蘑菇标签
unpair: ../yap/dumps/18429_raw.jpg
unpair: ../yap/dumps/19283_raw.jpg
unpair: ../yap/dumps/19328_raw.jpg
unpair: ../yap/text_dumps/21096_raw.jpg
unpair: ../yap/text_dumps/23793_raw.jpg
unpair: ../yap/text_dumps/24135_raw.jpg
unpair: ../yap/dumps3/115_沉重号角_raw.jpg
unpair: ../yap/dumps3/1765_祥箭_raw.jpg
unpair: ../yap/dumps4.0_tx/54_3_浊水的_raw.jpg
unpair: ../yap/dumps4.0_tx/56_2_浊水的一_raw.jpg
unpair: ../yap/dumps4.0_tx/66_1_出生的浊水_raw.jpg
'''

'''
unpair: ../yap/dumps4.0_tx4/498_2_「正义」的教_raw.jpg False
'''

'''
unpair: ../yap/dumps4.0_tx7/140_2_珊瑚真珠_raw.jpg False
'''

'''
dumps/15031_raw.jpg
dumps/15049_raw.jpg
text_dumps/1397_raw.jpg
dumps4.0_xs/107_2_蕈兽孢子_raw.jpg

dumps/17_raw.jpg
text_dumps/15279_raw.jpg
dumps4.0_tx/620_3_的_raw.jpg
'''

'''
dumps/563_raw.jpg
dumps/781_raw.jpg
dumps/914_raw.jpg
dumps/6703_raw.jpg
dumps/6791_raw.jpg
dumps/6660_raw.jpg
dumps/9399_raw.jpg
dumps/9400_raw.jpg
dumps/9423_raw.jpg
dumps/9401_raw.jpg
dumps/9402_raw.jpg
dumps/781_raw.jpg
dumps/9828_raw.jpg
dumps/1950_raw.jpg
text_dumps/20596_raw.jpg
text_dumps/2549_raw.jpg
text_dumps/20597_raw.jpg
text_dumps/20598_raw.jpg
dumps3/1889_教官的面叶_raw.jpg
dumps4.0_tx/764_2_冒险家_raw.jpg
dumps4.0_tx/999_2_冒险家_raw.jpg
dumps4.0_tx/1804_2_浊水的一_raw.jpg
dumps4.0_tx2/534_0_浊水的一_raw.jpg
dumps4.0_tx2/575_1_兽鸦印_raw.jpg
dumps4.0_tx3/153_1_铁_raw.jpg
text_dumps/8088_raw.jpg
text_dumps/8099_raw.jpg
text_dumps/8712_raw.jpg
text_dumps/9359_raw.jpg
text_dumps/9771_raw.jpg
text_dumps/10575_raw.jpg
text_dumps/10899_raw.jpg
text_dumps/13565_raw.jpg
text_dumps/13873_raw.jpg
text_dumps/16089_raw.jpg
text_dumps/21096_raw.jpg
text_dumps/21110_raw.jpg
text_dumps/23755_raw.jpg
text_dumps/27649_raw.jpg
text_dumps/33216_raw.jpg
text_dumps/34326_raw.jpg
dumps3/960_日人果_raw.jpg
dumps3/1155_破损的面具_raw.jpg
dumps3/1339_破损的面具_raw.jpg
dumps3/1407_归兽鸦_raw.jpg
dumps3/2046_孢_raw.jpg
dumps3/2205_固者_raw.jpg
dumps3/2295_来自的处_raw.jpg
dumps4.0_tx/409_3_的洛_raw.jpg
dumps4.0_tx/412_3_的_raw.jpg
dumps4.0_tx/469_2_枯的一的_raw.jpg
dumps4.0_tx/772_1_游医的枭孢_raw.jpg
dumps4.0_tx/1113_3_游医的方巾_raw.jpg
dumps4.0_tx/1800_2_浊水的一_raw.jpg
dumps4.0_tx/1804_2_浊水的一_raw.jpg
dumps4.0_tx2/150_3_地脉的新_raw.jpg
dumps4.0_tx2/155_2_史莱姆凝液_raw.jpg
dumps4.0_tx2/575_1_兽鸦印_raw.jpg
dumps4.0_tx2/589_2_浊水的一_raw.jpg
dumps4.0_tx2/617_2_浊水的一掬_raw.jpg
dumps4.0_tx3/48_1_号_raw.jpg
dumps4.0_tx3/153_1_铁_raw.jpg
dumps4.0_tx3/185_4_混沌回路_raw.jpg
dumps4.0_tx3/362_4_击_raw.jpg
dumps4.0_tx4/185_2_战狂的时计_raw.jpg
dumps4.0_tx4/325_1_混沌核_raw.jpg
dumps4.0_tx4/408_3_兽肉_raw.jpg
dumps4.0_tx4/509_2_大英雄的_raw.jpg
dumps4.0_tx4/1134_2_沉重号角_raw.jpg
dumps4.0_tx4/1155_2_混沌回路_raw.jpg
dumps4.0_tx7/141_3_珊瑚真珠_raw.jpg
dumps4.0_syfs/170_1_结地破的旧枝_raw.jpg
dumps4.0_xs/37_2_蕈兽孢子_raw.jpg
dumps4.0_longx/125_2_何人珍藏放_raw.jpg
dumps4.0_longx/162_3_来自何处的放之花_raw.jpg
dumps4.0_longx/200_4_何」淡通之_raw.jpg
dumps4.0_longx/442_1_地野的枯_raw.jpg
dumps4.0_longx/1003_4_士官的_raw.jpg
dumps4.0_longx/1128_2_异界晶命_raw.jpg
dumps4.0_longx/1672_1_藏的印_raw.jpg
'''

'''
dumps/10794_raw.jpg
dumps/11532_raw.jpg
text_dumps/9341_raw.jpg
dumps4.0_tx/157_2_浊水的一滴_raw.jpg
dumps4.0_tx/666_2_芭_raw.jpg
dumps4.0_tx3/493_1_地脉的旧_raw.jpg
dumps4.0_tx3/992_1_精关正齿轮_raw.jpg
dumps4.0_syfs/250_3_地脉的旧枝_raw.jpg
dumps4.0_xs/2_1_蕈兽孢_raw.jpg
dumps4.0_yjls/60_2_混沌容置_raw.jpg
dumps4.0_longx/1125_4_隙间之_raw.jpg
'''

'''
text_dumps/2555_raw.jpg

text_dumps/32605_raw.jpg


dumps/3751_raw.jpg
dumps/4623_raw.jpg
dumps/10811_raw.jpg
text_dumps/16630_raw.jpg
text_dumps/18909_raw.jpg
text_dumps/19003_raw.jpg
text_dumps/20896_raw.jpg
dumps4.0_tx3/899_2_须彩蔷薇_raw.jpg
dumps4.0_tx5/176_3_混沌机眼_raw.jpg
dumps4.0_longx/556_3_地脉的枯_raw.jpg
'''
