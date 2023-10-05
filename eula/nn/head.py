import torch
import torch.nn as nn

class MobileNetV3Small_head(nn.Module):
    def __init__(self, num_classes, channel, bn_momentum=0.1) -> None:
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        cls = self.cls_head(x)
        offset = self.reg_head(x)
        return cls, offset