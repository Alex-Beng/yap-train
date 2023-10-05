import torch


from eula.nn.backbone import MobileNetV3Small
from eula.nn.head import MobileNetV3Small_head
from eula.nn.centernet import CenterNet_MobilenetV3Small

if __name__ == "__main__":
    model = CenterNet_MobilenetV3Small(num_classes=10)
    model._init_weights()
    x = torch.randn(1, 3, 64, 384)
    cls, offset = model(x)
    print(cls.shape, offset.shape)