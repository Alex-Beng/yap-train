# use this to train
# cause the train.py is used for mona
import sys
import pathlib
import os

import torch
import cv2

from datagen.datagen import EulaDataset
from config import config
from train import train, test

# print("hello world")

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        train_size = config["train_size"]
        validate_size = config["validate_size"]

        folder = pathlib.Path(__file__).parent.parent / "data" / "eula"
        
        if not folder.is_dir():
            folder.mkdir(parents=True)
        # TODO: update background images
        bg_imgs = [cv2.imread("./assets/test.png")]
        dataset = EulaDataset(2, train_size, (384, 64), bg_imgs)
        train_datas = [dataset[i] for i in range(train_size)]
        torch.save(train_datas, folder / "train.pt")
        dataset = EulaDataset(2, validate_size, (384, 64), bg_imgs)
        validate_datas = [dataset[i] for i in range(validate_size)]
        torch.save(validate_datas, folder / "validate.pt")
        
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
