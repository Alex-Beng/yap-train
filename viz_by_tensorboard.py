import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from mona.nn.model2 import Model2
from mona.text import word_to_index, index_to_word


model = Model2(len(word_to_index), in_channels=1)
dummy_input = torch.rand(1, 1, 32, 384)

writer = SummaryWriter("runs/model2")
writer.add_graph(model, dummy_input)
writer.close()
