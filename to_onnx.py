import torch
import torchvision.transforms as T
from torchsummary import summary
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import json

from mona.nn.model import Model
from mona.nn.model2 import Model2
from mona.text import word_to_index, index_to_word
from mona.datagen.datagen import generate_pure_bg_image,generate_pickup_image, generate_mix_image


# TODO: make path configurable
model_folder = "models"
name = "model_training.pt"
name = "model_best.pt"
# name = "model_acc100-epoch860.pt"
# name = "model_acc9999-epoch1.pt"
onnx_name = name.rsplit(".", 2)[0] + ".onnx"
onnx_name = "model_training.onnx"
# net = Model(len(word_to_index))
net = Model2(len(word_to_index), in_channels=1)

device = "cuda" if torch.cuda.is_available() else "cpu"

net.load_state_dict(torch.load(f"{model_folder}/{name}", map_location=torch.device(device)))
net.eval()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(net))
# exit()
# summary(net, (1, 32, 384))


to_tensor = T.ToTensor()
x, label = generate_pickup_image()
x2, label2 = generate_pickup_image()
x = to_tensor(x)
x2 = to_tensor(x2)
x = torch.stack([x, x2, x, x, x], dim=0)

print(x.shape)
# x.unsqueeze_(0) # (1, 3, 32, width)
y = net(x)      # (width / 8, 1, lexicon_size)
# y3 = net(x3)
print(x.size(), y.size())

torch.onnx.export(
    net,
    x,
    f"{model_folder}/{onnx_name}",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {3: "image_width", 0: "image_num"},
        "output": {0: "seq_length"},
    }
)

onnx_model = onnx.load(f"{model_folder}/{onnx_name}")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(f"{model_folder}/{onnx_name}")


# print(ort_session.get_inputs()[0])
ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}

import time
beg_time = time.time()
# for _ in range(10000):
ort_outs = ort_session.run(None, ort_inputs)

print(f"onnxruntime inference time: {time.time() - beg_time}")
# print()
# print(ort_outs[0].shape)
np.testing.assert_allclose(y.detach().numpy(), ort_outs[0], rtol=1e-3, atol=1e-5)

with open(f"{model_folder}/index_2_word.json", "w", encoding="utf-8") as f:
    j = json.dumps(index_to_word, indent=4, ensure_ascii=False).encode("utf8")
    f.write(j.decode())
