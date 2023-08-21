# 使用合成数据训练好的模型来清理数据
import random
import os
from copy import deepcopy

from PIL import Image, ImageFont, ImageDraw
import numpy as np

from mona.text import lexicon, index_to_word
from mona.nn.model2 import Model2

from mona.nn import predict as predict_net

import cv2
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import argparse


def js_dp(obj, path):
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

def js_ld(path):
    return json.load(open(path, 'r', encoding='utf-8'))


genshin_x = js_ld('../yap/xx.json')
genshin_y = js_ld('../yap/yy.json')

root_path = "../yap/"
genshin_n = len(genshin_x)

parser = argparse.ArgumentParser(
        description='validate data using a model')
parser.add_argument('model_file', type=str,
                    help='The model file. e.g. model_training.pt')
args = parser.parse_args()
model_file_path = args.model_file

device = "cpu"
net = Model2(len(index_to_word), 1).to(device)
net.load_state_dict(torch.load(
        model_file_path, map_location=torch.device(device)))
net.eval()
with torch.no_grad():
    for i in range(genshin_n):
        path = os.path.join(root_path, genshin_x[i])
        with Image.open(path) as img:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            img = cv2.resize(img, (145, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
            img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=255)
            img_cv = deepcopy(img)
            img = Image.fromarray(img)
            
            im = transforms.ToTensor()(img)
            # 变为 (1, 1, 32, 384)
            im = im.unsqueeze(0)
            im = im.to(device)
            predict = predict_net(net, im)[0]
            if predict != genshin_y[i]:
                print(f"unpair: {path} {predict==genshin_y[i]}\n =={predict}== vs =={genshin_y[i]}==")
                cv2.imshow("unpair", img_cv)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    exit()

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