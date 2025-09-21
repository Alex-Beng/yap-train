'''
用于 ctc center loss 获得初始的中心。
FYI: https://github.com/PaddlePaddle/PaddleOCR/blob/7069e78b0a737bbdea61be68c496f15d2539a73f/docs/version2.x/ppocr/blog/enhanced_ctc_loss.md
'''
import os
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tqdm
from IPython import embed

from mona.text import index_to_word, word_to_index
from mona.nn.model import Model
# from mona.nn.model2 import Model2
from mona.nn.model2_mb import Model2
from mona.datagen.datagen import generate_pure_bg_image, generate_pickup_image, random_text, random_text_genshin_distribute, generate_mix_image
from mona.config import mb_config as config
from mona.nn import predict as predict_net

import numpy as np
from PIL import Image

import sys
import time
import random
import pickle
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
is_gpu = device != 'cpu'

FEATS = {}
def hook_fn(module, input, output):
    FEATS['in'] = input[0] # is tuple
    FEATS['out'] = output


class MyOnlineDataSet(Dataset):
    def __init__(self, size: int):
        self.size = size
        self.hard_words = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # im, text = generate_pickup_image(random_text_genshin_distribute)
        if  not len(self.hard_words):
            im, text = generate_mix_image(random_text_genshin_distribute, config['data_genshin_ratios'], config['pickup_ratio'])
        else:
            im, text = generate_mix_image(self.rand_fn, config['data_genshin_ratios'], config['pickup_ratio'])
        tensor = transforms.ToTensor()(im)
        text = text.strip()
        return tensor, text
    
    # hook 掉rand func
    def set_hard_word(self, words):
        self.hard_words = words
    
    def rand_fn(self):
        return ''.join(random.choices(self.hard_words, k=random.randint(1, 2)))


if __name__ == "__main__":
    # crnn
    # net = Model(len(index_to_word)).to(device)
    # svtr
    net = Model2(len(index_to_word), 1, hidden_channels=384).to(device)
    # net = Model2(len(index_to_word), 1, hidden_channels=128, num_heads=4).to(device)

    parser = argparse.ArgumentParser(
        description='Validate a model using online generated data from datagen')
    parser.add_argument('model_file', type=str,
                        help='The model file. e.g. model_training.pt')
    # 是否保存错误的样本到训练文件夹
    parser.add_argument('--no_save_err', action='store_true',
                        help='Save the incorrect samples to the training folder')
    save_err = not parser.parse_args().no_save_err

    args = parser.parse_args()
    model_file_path = args.model_file
    model_file_path = model_file_path.replace("\\", "/")
    model_file_name = model_file_path.split("/")[-1].split(".")[0]

    print(f"Validating {model_file_path}")
    net.load_state_dict(torch.load(
        model_file_path, map_location=torch.device(device)))

    batch_size = 32
    max_plot_incorrect_sample = 100
    num_samples = 1000000

    validate_dataset = MyOnlineDataSet(num_samples)
    validate_loader = DataLoader(
        validate_dataset, batch_size=batch_size, num_workers=config["dataloader_workers"])

    net.eval()
    net.linear2.register_forward_hook(hook_fn)

    err = 0
    total = 0
    last_time = time.time()
    yy_mm_dd_hh_mm_ss = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    char_center = {}
    with torch.no_grad():
        for x, label in validate_loader:
            x = x.to(device)
            # print(label)
            # embed()
            predict = predict_net(net, x, True)
            feats = FEATS['in']

            for i_sample in range(len(label)):
                pred, logit = predict[i_sample]
                truth = label[i_sample]
                feat = feats[i_sample].numpy() if not is_gpu else feats[i_sample].cpu().numpy()
                
                # embed()
                # exit()
                # stolen from pp-ocr
                if pred == truth:
                    for idx_time in range(len(logit)):
                        word = logit[idx_time]
                        index = word_to_index[word]
                        if index in char_center:
                            char_center[index][0] = (
                                char_center[index][0] * char_center[index][1] + feat[idx_time]
                            ) / char_center[index][1]
                            char_center[index][1] += 1
                        else:
                            char_center[index] = [feat[idx_time], 1]
            # Stats
            print(f"conv rate: {len(char_center)} / {len(index_to_word)} ")
            # get the words
            left_idexs = set(index_to_word.keys()) - set(char_center.keys())
            left_words = [index_to_word[i] for i in left_idexs]
            if len(left_words) < 300:
                validate_loader.dataset.set_hard_word(left_words)
                print(left_words)

            if len(left_words) == 0 and min([p[1] for p in char_center]) > 10:
                break

        # embed()
        # serialize to disk
        with open("train_center.pkl", "wb") as f:
            pickle.dump(char_center, f)



['枚', '铺', '蚀', '赠', '衍', '载', '俐', '辖', '阵', '麟', '匙', '玩', '协', '棠', '，', '兆', '设', '堡', '炎', '蒸', '掣', '飘', '仕', '付', '吏', '击', '钥', '汽']