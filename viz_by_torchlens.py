import torch
import torchvision
import torchlens as tl


from mona.nn.model2 import Model2
from mona.text import word_to_index, index_to_word


model = Model2(len(word_to_index), in_channels=1)
dummy_input = torch.rand(1, 1, 32, 384)

model_history = tl.log_forward_pass(model, dummy_input, layers_to_save='all', vis_opt='unrolled')
print(model_history)

