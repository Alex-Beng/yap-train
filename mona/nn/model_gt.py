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


# 从detr超的
# 用于当头
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# batch, 3, 224, 224 -> batch, 2/4/360
class Model_GT(nn.Module):
    def __init__(self, backbone_name='resnet50', depth=0, hidden_channels=192, num_heads=8, out_size=2, cls_head=False):
        super(Model_GT, self).__init__()
        if backbone_name == 'mobile':
            self.cnn = MobileNetV3Small_GT(out_size=hidden_channels, in_channels=3)
            seq_len = 24
        elif 'resnet' in backbone_name:
            resnet = getattr(torchvision.models, backbone_name)(pretrained=True)
            num_channels = 512 if backbone_name in ('resnet18', 'resnet34') else 2048
            self.cnn = nn.Sequential(*list(resnet.children())[:-2], nn.Conv2d(num_channels, hidden_channels, 1))
            seq_len = 49
        
        # use flatten, 7, 7 -> 49
        # the same flatten as DETR
        self.pe = PositionalEncoding(dim=hidden_channels, length=seq_len)
        self.query_pos = nn.Parameter(torch.rand(seq_len, hidden_channels))

        # 添加一个batchnorm
        # self.bm = nn.BatchNorm1d(49)
        # 添加一个dropout
        self.dp = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        # self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.blocks = nn.Sequential()
        for i in range(depth):
            # 这个mixing block实际就是个transformer block
            # 直接无痛切换到transformer
            # block = MixingBlock(
            #     dim=hidden_channels,
            #     num_heads=num_heads,
            #     is_local=False,
            #     drop_path=0.0,
            #     hw=None,
            #     input_length=seq_len,
            #     mlp_ratio=2,
            #     attention_drop=0.1,
            #     drop=0.1,
            # )
            block = nn.Transformer(
                d_model=hidden_channels,
                nhead=num_heads,
                num_encoder_layers=8,
                num_decoder_layers=6,
                dim_feedforward=hidden_channels,
                dropout=0.1,
                activation='relu',
                batch_first=False
            )
            self.blocks.add_module(f"mix{i}", block)
        self.norm = nn.LayerNorm(hidden_channels)
        # 变成回归头，而不是分类头
        # 0 for the MAN, 1 for the VEIW
        # TODO: make it not magic number
        # TODO: 测试更多的回归头的方法
        # self.linear2 = nn.Linear(hidden_channels * 49, out_size)
        # self.reg_head = nn.Linear(hidden_channels * seq_len, out_size)
        self.reg_head = MLP(hidden_channels * seq_len, hidden_channels, out_size, 3)
        # self.cls_head = nn.Linear(hidden_channels * seq_len, 360)
        self.cls_head = MLP(hidden_channels * seq_len, hidden_channels, 360, 3)
        self.use_cls_head = cls_head

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        # exit()
        x = x.flatten(2)
        # x = x.permute((2, 0, 1))
        x = x.permute((0, 2, 1))
        # print(x.shape)
        x = self.pe(x)
        x = x.permute((1, 0, 2))
        # x = self.bm(x)
        x = self.dp(x)
        # x = self.linear1(x)
        # x = self.dp2(x)
        # x = self.dp3(x)
        # x = self.blocks(x, self.query_pos)
        for block in self.blocks:
            # 需要把query_pos的batch维度扩展到和x一样
            _query_pos = self.query_pos.unsqueeze(1).repeat(1, x.shape[1], 1)
            # _query_pos = self.query_pos.unsqueeze(1)
            # print(x.shape, _query_pos.shape)
            x = block(x, _query_pos)
            # print(x.shape)

        x = self.norm(x)
        x = x.permute((1, 0, 2)).flatten(1, 2)
        # print(x.shape)

        if self.use_cls_head:
            x = self.cls_head(x)
        else:
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

