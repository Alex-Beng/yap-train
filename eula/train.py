import datetime
import os

import cv2
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

from nn.centernet import CenterNet_MobilenetV3Small
from nn.loss import focal_loss, reg_l1_loss
from datagen.datagen import EulaDataset
from config import config

device = config["device"]
device = "cpu" if not torch.cuda.is_available() else device

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def load_bg_imgs():
    path = "../yap/dumps_full/"
    # 获取文件夹下所有图片
    files = os.listdir(path)
    # 读取图片
    imgs = []
    for file in files:
        imgs.append(cv2.imread(path + file))
    return imgs


def train():
    loss_log_file = open("loss_log.txt", "a+")

    net = CenterNet_MobilenetV3Small(config["num_classes"]).to(device)
    if config["pretrain"]:
        net.load_state_dict(
            torch.load(f'models/{config["pretrain_name"]}')
        )
    data_aug_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(1, 1),
                transforms.GaussianBlur(3, 3),
                transforms.GaussianBlur(5, 5),
                # transforms.GaussianBlur(7,7),
            ])], p=0.5),

        # transforms.RandomApply([
        #     transforms.RandomCrop(size=(384, 64)),
        #     transforms.Resize((384, 64)),
        #     ], p=0.5),

        transforms.RandomApply([AddGaussianNoise(mean=0, std=1/255)], p=0.5),
    ])

    bg_imgs = load_bg_imgs()
    train_dataset = EulaDataset(
        config["num_classes"], config["train_size"], (384, 64), bg_imgs) \
        if config["online_train"] else torch.load("data/eula/train.pt")
    validate_dataset = EulaDataset(
        config["num_classes"], config["validate_size"], (384, 64), bg_imgs) \
        if config["online_val"] else torch.load("data/eula/validate.pt")
    
    train_loader = DataLoader(train_dataset, shuffle=True, 
        num_workers=config["dataloader_workers"], batch_size=config["batch_size"],)
    validate_loader = DataLoader(validate_dataset, shuffle=True,
        num_workers=config["dataloader_workers"], batch_size=config["batch_size"],)
    
    optimizer = optim.Adadelta(net.parameters())

    epoch = config["epoch"]
    print_per = config["print_per"]
    save_per = config["save_per"]
    batch_cnt = 0
    start_time = datetime.datetime.now()

    curr_best_loss = 1000000000
    for epoch in range(config['init_epoch'], epoch):
        for batch in train_loader:
            optimizer.zero_grad()

            batch_images, batch_hms, batch_regs, batch_reg_msks = batch
            batch_images = batch_images.to(device)
            batch_hms = batch_hms.to(device)
            batch_regs = batch_regs.to(device)
            batch_reg_msks = batch_reg_msks.to(device)

            batch_images = data_aug_transform(batch_images)

            batch_size = batch_images.size(0)

            batch_pred = net(batch_images)
            batch_hms_pred, batch_regs_pred = batch_pred

            c_loss = focal_loss(batch_hms_pred, batch_hms)
            off_loss = reg_l1_loss(batch_regs_pred, batch_regs, batch_reg_msks)
            # print(c_loss, off_loss)
            loss =  c_loss + off_loss
            # loss = c_loss

            loss.backward()
            optimizer.step()

            cur_time = datetime.datetime.now()
            if batch_cnt % print_per == 0 and batch_cnt != 0:
                tput = batch_size * batch_cnt / (cur_time - start_time).total_seconds()
                print(f"Epoch {epoch}, Batch {batch_cnt}, Loss {loss:.8f}, tput: {tput:.8f}, c_loss_rate: {c_loss / loss:.8f}")
            
            if batch_cnt % save_per == 0 and batch_cnt != 0:
                torch.save(net.state_dict(), f"models/eula/model_training.pt")
                print("Model saved")
                if loss < curr_best_loss:
                    torch.save(net.state_dict(), f"models/eula/model_best.pt")

                    # refresh the file
                    
                    loss_log_file = open("loss_log.txt", "a+") if loss_log_file.closed else loss_log_file
                    print(f"Model saved with loss: {loss}", file=loss_log_file)
                    loss_log_file.close()
                    
                    curr_best_loss = loss
            batch_cnt += 1


def test(model_path: str = "models/eula/model_training.pt"):
    # only use eye to test
    net = CenterNet_MobilenetV3Small(config["num_classes"]).to(device)
    net.load_state_dict(
        torch.load(model_path)
    )

    bg_imgs = load_bg_imgs()
    validate_dataset = EulaDataset(
        config["num_classes"], config["validate_size"], (384, 64), bg_imgs) \
        if config["online_val"] else torch.load("data/eula/validate.pt")
    validate_loader = DataLoader(validate_dataset, shuffle=False,
        num_workers=config["dataloader_workers"], batch_size=1, )
    
    for batch in validate_loader:
        img, hm, reg, reg_mask = batch
        img = img.to(device)

        with torch.no_grad():
            hm_pred, reg_pred = net(img)

        hm_pred = hm_pred.cpu().numpy()
        img = img.cpu().numpy()

        img *= 255
        img = np.array(img).reshape(3, 384, 64)
        print(img.shape)
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        
        hm_pred *= 255
        hm_pred = np.array(hm_pred).reshape(96, 16, 2)[:, :, 1]

        hm_pred = cv2.resize(hm_pred, (64, 384))

        cv2.imshow("img", img)
        cv2.imshow("hm", hm_pred)
        cv2.waitKey()



