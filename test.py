import torch

from mona.nn.svtr import Attention, MixingBlock, PatchEmbed, SVTRNet
from mona.nn.model import Model
from mona.nn.mobile_net_v3 import MobileNetV3Small_GT, MobileNetV3Small
from mona.nn.model2 import Model2
from mona.nn.model_gt import Model_GT

lexicon_size = 400

# net = Attention(
#     dim=128,
#     num_heads=4,
#     is_local=True,
#     hw=[8, 8],
#     dropout=0.1
# )
#
# x = torch.randn(1, 64, 128)
# y = net(x)
# # y = torch.softmax(y, dim=2)
# print(y)
# print(y.size())

# net = MixingBlock(
#     dim=128,
#     num_heads=8,
#     is_local=False,
#     hw=(8, 8)
# )
# x = torch.randn(1, 64, 128)
# y = net(x)
# print(y.shape)

# net = PatchEmbed(
#     img_size=(32, 256),
#     in_channels=3,
#     embed_dim=64
# )
# x = torch.randn(1, 3, 32, 256)
# y = net(x)
# print(y.shape)

# net = SVTRNet(
#     img_size=(32, 384),
#     in_channels=1,
#     out_channels=256
# )
# # net = Model(100)
# x = torch.randn(2, 1, 32, 384)
# y = net(x)
# print(y)


# net = SVTRNet(
#     in_channels=256,
#     in_length=24,
#     out_channels=lexicon_size,
#     hidden_channels=120,
#     depth=2,
#     num_heads=8
# )


# cnn = MobileNetV3Small_GT(out_size=256, in_channels=1)
# # # net = MobileNetV3Small(out_size=64, in_channels=1)
# # # net = Model(100)
# x = torch.randn(1, 1, 224, 224)
# x = cnn(x)
# print(x.shape)
# # x = x.squeeze()
# x = x.flatten(2)
# print(x.shape)
# x = x.permute((0, 2, 1))
# # x = net(x)
# print(x.shape)
# # exit()


# net = Model2(lexicon_size=lexicon_size, in_channels=1)
net = Model_GT(1)
# # net = Model(lexicon_size)
x = torch.randn(2, 1, 32, 384)
x = torch.randn(10, 1, 224, 224)
x = net(x)
print(x.shape)
exit()
# print(x)

# just test input and output size
model = Model_GT(1)

img = torch.randn(1, 1, 224, 224)
output = model(img)
print(output.size())