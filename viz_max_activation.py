# 冻结网络权重，训练输入图片，获得最大激活值/最小ctc loss的图片
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2

from mona.text import index_to_word, word_to_index
from mona.nn.model import Model
from mona.nn.svtr import SVTRNet
from mona.datagen.datagen import generate_pure_bg_image, generate_pickup_image, random_text, random_text_genshin_distribute, generate_mix_image
from mona.config import maxact_config as config
from mona.nn import predict as predict_net
from mona.nn.model2_mb import Model2
from typing import List

import datetime
from time import sleep

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = config["device"]
val_cnt = 0

def get_model_from_pretrained():
    net = Model2(len(index_to_word), 1, depth=2, hidden_channels=384, backbone_name='mobile').to(device)
    net.load_can_load(torch.load(f"models/{config['pretrain_name']}", map_location=device))
    # freeze all layers
    for param in net.parameters():
        param.requires_grad_(False)
    return net

def get_input_tensor(bat_size=1, method="zero", ts: 'List[str]'=[]):
    # return a tensor of shape (1, 1, 32, 384) with requires_grad=True

    if method == "rand":
        ts = torch.rand(bat_size, 1, 32, 384, device=device) * 255
    elif method == "zero":
        ts = torch.zeros(bat_size, 1, 32, 384, device=device)
    elif method == "datagen":
        assert len(ts) == bat_size
        def fake_rand():
            cnt = -1
            def inner():
                nonlocal cnt
                cnt += 1
                return ts[cnt]
            return inner
        fk_rd_func = fake_rand()
        imgs = [generate_mix_image(fake_rand())[0] for _ in range(bat_size)]
        imgs = [transforms.ToTensor()(img) for img in imgs]
        ts = torch.stack(imgs)
        ts = ts.to(device)
    else:
        # TODO: use data gen to get the input tensor
        raise ValueError("method not supported")
    ts.requires_grad_(True)
    return ts

def get_target_strs(bat_size=1, method='rand') -> 'List[str]':
    if method == "rand":
        return [random_text_genshin_distribute() for _ in range(bat_size)]
    else:
        raise ValueError("method not supported")

def get_target(s: 'List[str]'):
    target_length = []

    target_size = 0
    for i, target in enumerate(s):
        target_length.append(len(target))
        target_size += len(target)

    target_vector = []
    for target in s:
        for char in target:
            index = word_to_index[char]
            if index == 0:
                print("error")
            target_vector.append(index)

    target_vector = torch.LongTensor(target_vector)
    target_length = torch.LongTensor(target_length)
    
    target_vector = target_vector.to(device)
    target_length = target_length.to(device)
    return target_vector, target_length


def main():
    bat_size = config["batch_size"]
    init_tensor_method = config["init_tensor_method"]
    init_str_method = config["init_str_method"]
    
    t_strs = get_target_strs(bat_size, method=init_str_method)

    net = get_model_from_pretrained()
    input_tensor = get_input_tensor(bat_size, method=init_tensor_method, ts=t_strs)
    targt_vector, target_length = get_target(t_strs)

    input_lengths = torch.full((bat_size,), 24, dtype=torch.long, device=device)

    print(input_tensor.shape)
    print(targt_vector.shape)
    print(target_length.shape)
    print(t_strs)

    # forward pass test
    with torch.no_grad():
        output = net(input_tensor)
        print(output.shape)

    # optimize the input tensor
    optimizer = optim.Adam([input_tensor], lr=config["lr"])
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)

    iters = config["iters"]
    show_per = config["show_per"]
    min_loss = float("inf")

    for it in range(iters):
        optimizer.zero_grad()
        output = net(input_tensor)

        loss = ctc_loss(output, targt_vector, input_lengths, target_length)

        loss.backward()
        optimizer.step()

        # print the loss
        print(f"\riter {it}, loss: {loss.item()}", end='')

        # 前100次迭代密集
        if it % show_per == 0 or it < 100:
            print(f"iter {it}")
            # TODO: concat the images and imshow
            # 输出input tensor 内的最大最小值
            print(input_tensor.max(), input_tensor.min())
            # make bat_size, 1, 32, 384 -> List[32, 384]
            imgs = [input_tensor[i, 0].detach().cpu().numpy() for i in range(bat_size)]
            # concat the images
            img = np.concatenate(imgs, axis=0)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            # imshow
            cv2.imshow("max activation", img)

            k = cv2.waitKey(20)
            if k == ord('q'):
                break
            cv2.imwrite(f"maxact/{it}.png", img)
    
def pics_to_video():
    root_path = "./maxact/"
    img_names = [f"{root_path}{i}.png" for i in range(0, 100)]
    img_names += [f"{root_path}{i}.png" for i in range(100, 69410, 10)]
    print(img_names[:2], img_names[-2:])

    for img_name in img_names:
        img = cv2.imread(img_name)
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # main()
    pics_to_video()