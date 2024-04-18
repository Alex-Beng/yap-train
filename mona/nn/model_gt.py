# regression head's model
# model_gt: gt means (回)归头，meaning regression head in Chinese
# 用于回归小地图的视角朝向

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from mona.nn.mobile_net_v3 import MobileNetV3Small_GT
from mona.nn.svtr import MixingBlock, SVTRNet, PositionalEncoding, SubSample
from mona.nn.mobile_net_v3 import MobileNetV3Block
from mona.text import index_to_word, word_to_index


class Model_GT(nn.Module):
    def __init__(self, in_channels, depth=2, hidden_channels=192, num_heads=8, out_size=2):
        super(Model_GT, self).__init__()
        # self.cnn = MobileNetV3Small_GT(out_size=hidden_channels, in_channels=in_channels)

        # cnn 切换为 resnet
        resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])


        # use flatten, 7, 7 -> 49
        self.pe = PositionalEncoding(dim=hidden_channels, length=49) 

        # 添加一个batchnorm
        self.bm = nn.BatchNorm1d(49)
        # 添加一个dropout
        self.dp = nn.Dropout(0.1)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.blocks = nn.Sequential()
        for i in range(depth):
            block = MixingBlock(
                dim=hidden_channels,
                num_heads=num_heads,
                is_local=False,
                drop_path=0.0,
                hw=None,
                input_length=49, # TODO: make it not magic number
                mlp_ratio=2,
                attention_drop=0.1,
                drop=0.1,
            )
            self.blocks.add_module(f"mix{i}", block)
        self.norm = nn.LayerNorm(hidden_channels)
        # 变成回归头，而不是分类头
        # 0 for the MAN, 1 for the VEIW
        # TODO: make it not magic number
        # TODO: 测试更多的回归头的方法
        # self.linear2 = nn.Linear(hidden_channels * 49, out_size)
        self.reg_head = nn.Linear(hidden_channels * 49, out_size)

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        # exit()
        x = x.flatten(2)
        x = x.permute((0, 2, 1))
        x = self.pe(x)
        x = self.bm(x)
        # x = self.dp(x)
        x = self.linear1(x)
        # x = self.dp(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.flatten(1)
        x = self.reg_head(x)
        # 无需更多的操作，相当于线性回归

        return x
    def load_can_load(self, pretrained_dict, old_idx2word_path="models/index_2_word.json"):
        try:
            self.load_state_dict(pretrained_dict, strict=False)
        except Exception as e:
            print("[warning] cannot load the model")

    def freeze_backbone(self):
        for param in self.cnn.parameters():
            param.requires_grad = False
    def unfreeze_backbone(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
