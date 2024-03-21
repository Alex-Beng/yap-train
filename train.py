import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from mona.text import index_to_word, word_to_index
from mona.nn.model import Model
from mona.nn.svtr import SVTRNet
from mona.datagen.datagen import generate_pure_bg_image, generate_pickup_image, random_text, random_text_genshin_distribute, generate_mix_image
from mona.config import config
from mona.nn import predict as predict_net
from mona.nn.model2 import Model2

import datetime
from time import sleep

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = config["device"]

# a list of target strings
def get_target(s):
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

    return target_vector, target_length


def validate(net, validate_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in validate_loader:
            x = x.to(device)
            predict = predict_net(net, x)
            # print(predict)
            correct += sum([1 if predict[i] == label[i] else 0 for i in range(len(label))])
            errs = [(predict[i], label[i]) for i in range(len(label)) if predict[i] != label[i] and not ( label[i][:7] == "尚需生长时间：" and predict[i][:7] == "尚需生长时间：")]
            if len(errs) < 5:
                print(errs)
            else:
                print(f'too many errors: {len(errs)}')
            total += len(label)

    net.train()
    return correct / total


def train():
    # net = Model(len(index_to_word)).to(device)
    # net = Model2(len(index_to_word), 1, hidden_channels=128, num_heads=4).to(device)

    # 这是现在在用的model
    # model2就是SVTR（非原版）
    net = Model2(len(index_to_word), 1).to(device)

    # net = SVTRNet(
    #     img_size=(32, 384),
    #     in_channels=1,
    #     out_channels=len(index_to_word)
    # ).to(device)
    if config["pretrain"]:
        # net.load_state_dict(torch.load(f"models/{config['pretrain_name']}", map_location=device))
        net.load_can_load(torch.load(f"models/{config['pretrain_name']}", map_location=device))

    data_aug_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(1, 1),
                # transforms.GaussianBlur(3, 3),
                # transforms.GaussianBlur(5, 5),
                # transforms.GaussianBlur(7,7),
            ])], p=0.5),

        transforms.RandomApply([
            transforms.RandomCrop(size=(31, 383)),
            transforms.Resize((32, 384), antialias=True),
            ], p=0.5),

        transforms.RandomApply([AddGaussianNoise(mean=0, std=1/255)], p=0.5),
    ])
    only_genshin = config['data_only_genshin']
    train_dataset = MyOnlineDataSet(config['train_size'], is_val=only_genshin,
                                    pk_ratio=config["pickup_ratio"],
                                     pk_genshin_ratio=config['data_genshin_ratio']) if config["online_train"] else MyDataSet(
        torch.load("data/train_x.pt"), torch.load("data/train_label.pt"))
    validate_dataset = MyOnlineDataSet(config['validate_size'], is_val=True, 
                                       pk_ratio=config["pickup_ratio"],
                                       pk_genshin_ratio=config['data_genshin_ratio']) if config["online_val"] else MyDataSet(
        torch.load("data/validate_x.pt"), torch.load("data/validate_label.pt"))

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=config["dataloader_workers"], batch_size=config["batch_size"],)
    validate_loader = DataLoader(validate_dataset, num_workers=config["dataloader_workers"], batch_size=config["batch_size"])

    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    # optimizer = optim.Adadelta(net.parameters())
    optimizer = optim.AdamW(net.parameters(), lr=config['lr'])
    # optimizer = optim.RMSprop(net.parameters())
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)

    epoch = config["epoch"]
    print_per = config["print_per"]
    save_per = config["save_per"]
    batch = 0
    start_time = datetime.datetime.now()
    if config["freeze_backbone"]:
        net.freeze_backbone()
    for epoch in range(epoch):
        if config["freeze_backbone"] and epoch == config["unfreeze_backbone_epoch"]:
            net.unfreeze_backbone()
        for x, label in train_loader:
            # sleep(10)
            optimizer.zero_grad()
            target_vector, target_lengths = get_target(label)
            target_vector, target_lengths = target_vector.to(device), target_lengths.to(device)
            x = x.to(device)

            # Data Augmentation in batch
            x = data_aug_transform(x)

            batch_size = x.size(0)

            y = net(x)

            input_lengths = torch.full((batch_size,), 24, device=device, dtype=torch.long)
            loss = ctc_loss(y, target_vector, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            cur_time = datetime.datetime.now()

            if batch % print_per == 0 and batch != 0:
                tput = batch_size * batch / (cur_time - start_time).total_seconds()
                print(f"{cur_time} e{epoch} #{batch} tput: {tput} loss: {loss.item()}")
                # print("sleeping for a while")
                # sleep(5)

            if batch % save_per == 0 and batch != 0:
                print("Validating and checkpointing")
                rate = validate(net, validate_loader)
                print(f"{cur_time} rate: {rate * 100}%")
                torch.save(net.state_dict(), f"models/model_training.pt")
                # torch.save(net.state_dict(), f"models/model_training_{batch+1}_acc{int(rate*10000)}.pt")
                if rate == 1:
                    torch.save(net.state_dict(), f"models/model_acc100-epoch{epoch}.pt")
                if int(rate*10000) >= 9999 and config["save_acc9999"]:
                    torch.save(net.state_dict(), f"models/model_acc9999-epoch{epoch}.pt")

            batch += 1

    for x, label in validate_loader:
        x = x.to(device)
        # predict = net.predict(x)
        predict = predict_net(net, x)
        print("predict:     ", predict[:10])
        print("ground truth:", label[:10])
        break


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
    def __init__(self, size: int, is_val: bool=False, pk_ratio: float=0.5, pk_genshin_ratio: float=0.5):
        self.size = size
        self.is_val = is_val
        if is_val:
            self.gen_func = random_text_genshin_distribute
        else:
            self.gen_func = random_text
        self.pk_rt = pk_ratio
        self.pk_g_rt = pk_genshin_ratio

            
        # 创建生成图像的协程
        # 纯纯负提升吞吐量
        # import asyncio
        # self.queue = asyncio.Queue()
        # self.loop = asyncio.get_event_loop()
        # async def generate_pure_bg_image(queue):
        #     while True:
        #         im, text = generate_pickup_image()
        #         # im = transforms.ToTensor()(im)
        #         while self.queue.qsize() > 600:
        #             await asyncio.sleep(0.1)
        #         await self.queue.put((im, text))
        # self.loop.create_task(generate_pure_bg_image(self.queue))
    def get_xy(self):
        loop = self.loop
        im, text = loop.run_until_complete(self.queue.get())
        return im, text
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Generate data online
        # im, text = generate_pickup_image(self.gen_func, self.pk_g_rt)
        im, text = generate_mix_image(self.gen_func, self.pk_g_rt, self.pk_rt)
        # im, text = self.get_xy()
        im = transforms.ToTensor()(im)
        text = text.strip()

        return im, text
