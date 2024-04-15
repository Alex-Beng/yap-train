import sys
import os
import pathlib

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# from mona.text.stat import random_stat, random_value
from mona.datagen.datagen import generate_pure_bg_image, generate_image_sample, generate_pickup_image, random_text_genshin_distribute, generate_ui_image, generate_mix_image
from train import train
from gt_train import train as gt_train
from mona.config import config


import datetime

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        train_size = config["train_size"]
        validate_size = config["validate_size"]
        pk_ratio = config["pickup_ratio"]
        data_genshin_ratio = config["data_genshin_ratio"]

        folder = pathlib.Path("data")
        if not folder.is_dir():
            os.mkdir(folder)

        x = []
        y = []
        cnt = 0
        for _ in range(train_size):
            # im, text = generate_pure_bg_image()
            # im, text = generate_pickup_image()
            im, text = generate_mix_image(random_text_genshin_distribute, data_genshin_ratio, pk_ratio)
            tensor = transforms.ToTensor()(im)
            tensor = torch.unsqueeze(tensor, dim=0)
            x.append(tensor)
            y.append(text)

            cnt += 1
            if cnt % 1000 == 0:
                print(f'{cnt} / {train_size} {datetime.datetime.now()}')

        xx = torch.cat(x, dim=0)
        torch.save(xx, "data/train_x.pt")
        torch.save(y, "data/train_label.pt")

        x = []
        y = []
        for _ in range(validate_size):
            # im, text = generate_pure_bg_image()
            # im, text = generate_pickup_image()
            im, text = generate_mix_image()
            tensor = transforms.ToTensor()(im)
            tensor = torch.unsqueeze(tensor, dim=0)
            x.append(tensor)
            y.append(text)
            # if cnt % 1000 == 0:
            #     print(f'vali: {cnt} / {validate_size} {datetime.datetime.now()}')

        xx = torch.cat(x, dim=0)
        torch.save(xx, "data/validate_x.pt")
        torch.save(y, "data/validate_label.pt")

        # generate sample
        for i in range(50):
            # im, text = generate_pickup_image()
            im, text = generate_mix_image()
            im.save(f"data/sample_{i}.png")
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "gt_train":
        gt_train()

    elif sys.argv[1] == 'sample':
        folder = pathlib.Path("samples")
        if not folder.is_dir():
            os.mkdir(folder)

        for i in range(100):
            # im, text = generate_pickup_image(random_text_genshin_distribute, 0)
            im, text = generate_mix_image(random_text_genshin_distribute, 0)
            text = text.replace("/", "_")
            text = text.replace("?", "_")
            text = text.replace(":", "_")
            im.save(f"samples/{i}_{text}.png")
            # img_processed.save((f"samples/{i}_p.png"))
    elif sys.argv[1] == 'sample2':
        folder = pathlib.Path("samples2")
        if not folder.is_dir():
            os.mkdir(folder)

        for i in range(100):
            im, text = generate_image_sample()
            im.save(f"samples2/{i}.png")
    elif sys.argv[1] == 'sample3':
        folder = pathlib.Path("samples3")
        if not folder.is_dir():
            os.mkdir(folder)

        for i in range(100):
            im, text = generate_ui_image()
            im.save(f"samples3/{i}.png")