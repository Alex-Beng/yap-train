# make it train guitou

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn import functional as F

from mona.datagen.gt_datagen import generate_image
from mona.config import gt_config as config
from mona.nn.model_gt import Model_GT

import datetime
from time import sleep

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = config["device"]


def validate(net, validate_loader):
    net.eval()
    with torch.no_grad():
        for x, label in validate_loader:
            x = x.to(device)
            y_hat = net(x)
            label = label.to(device)
            # calculate the L1 loss
            loss = F.l1_loss(y_hat, label, reduction="mean")
            # print(f"Validation loss: {loss.item()}")
    net.train()
    return loss.item()


def train():
    net = Model_GT(3, out_size=4).to(device)


    if config["pretrain"]:
        # net.load_state_dict(torch.load(f"models/{config['pretrain_name']}", map_location=device))
        net.load_can_load(torch.load(f"models/gt/{config['pretrain_name']}", map_location=device))

    data_aug_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(1, 1),
                # transforms.GaussianBlur(3, 3),
                # transforms.GaussianBlur(5, 5),
                # transforms.GaussianBlur(7,7),
            ])], p=0.5),

        transforms.RandomApply([
            transforms.RandomCrop(size=(config['side_len']-2, config['side_len']-2)),
            transforms.Resize((config['side_len'], config['side_len']), antialias=True),
            ], p=0.5),

        transforms.RandomApply([
            transforms.RandomCrop(size=(config['side_len']-20, config['side_len']-20)),
            transforms.Resize((config['side_len'], config['side_len']), antialias=True),
            ], p=0.5),

        transforms.RandomApply([AddGaussianNoise(mean=0, std=10)], p=0.5),
    ])
    train_dataset = MyOnlineDataSet(config['train_size']) if config["online_train"] \
        else MyDataSet(torch.load("data/train_x.pt"), torch.load("data/train_label.pt"))
    validate_dataset = MyOnlineDataSet(config['validate_size']) if config["online_val"] \
        else MyDataSet(torch.load("data/validate_x.pt"), torch.load("data/validate_label.pt"))

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=config["dataloader_workers"], batch_size=config["batch_size"],)
    validate_loader = DataLoader(validate_dataset, num_workers=config["dataloader_workers"], batch_size=config["batch_size"])

    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    # optimizer = optim.Adadelta(net.parameters())
    optimizer = optim.AdamW(net.parameters(), lr=config['lr'])
    # optimizer = optim.RMSprop(net.parameters())

    epoch = config["epoch"]
    print_per = config["print_per"]
    save_per = config["save_per"]
    batch = 0
    # 回归任务也只能用loss而不是acc了吧？
    # 是不是1°之内就可以认为分类成功？
    curr_best_loss = float("inf")
    start_time = datetime.datetime.now()
    if config["freeze_backbone"]:
        net.freeze_backbone()
    for epoch in range(epoch):
        if config["freeze_backbone"] and epoch == config["unfreeze_backbone_epoch"]:
            net.unfreeze_backbone()
        for x, label in train_loader:
            # sleep(10)
            optimizer.zero_grad()
            target_vector = label
            target_vector = target_vector.to(device)
            x = x.to(device)

            # Data Augmentation in batch
            x = data_aug_transform(x)

            batch_size = x.size(0)

            y = net(x)

            loss = F.l1_loss(y, target_vector, reduction="mean")
            # 添加正则化loss
            # loss += 0.0001 * torch.norm(net.linear2.weight, p=2)
            loss.backward()
            optimizer.step()

            cur_time = datetime.datetime.now()

            if batch % print_per == 0 and batch != 0:
                tput = batch_size * batch / (cur_time - start_time).total_seconds()
                print(f"{cur_time} e{epoch} #{batch} tput: {tput:.2f} loss: {loss.item()}")
                # print("sleeping for a while")
                # sleep(5)

            if batch % save_per == 0 and batch != 0:
                print(f"curr best loss: {curr_best_loss}")
                print("Validating and checkpointing")
                val_loss = validate(net, validate_loader)
                print(f"{cur_time} loss: {val_loss}")
                torch.save(net.state_dict(), f"models/gt/model_training.pt")
                # torch.save(net.state_dict(), f"models/model_training_{batch+1}_acc{int(rate*10000)}.pt")
                if val_loss < curr_best_loss:
                    torch.save(net.state_dict(), f"models/gt/model_best.pt")
                    curr_best_loss = val_loss

            batch += 1


class MyDataSet(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.x[index]
        label = self.labels[index]

        return x, label

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MyOnlineDataSet(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Generate data online
        im, label = generate_image()
        # im, text = self.get_xy()
        im = transforms.ToTensor()(im)

        return im, label
