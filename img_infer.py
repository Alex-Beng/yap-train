# 输入图片，进行推理，返回推理结果

import torch

from mona.text import index_to_word
from mona.nn.model2 import Model2
from mona.nn import predict as predict_net, arr_to_string

import onnx
import onnxruntime

import numpy as np
from PIL import Image
import cv2

import sys
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    
    parser = argparse.ArgumentParser(
            description='Validate a model using online generated data from datagen')
    parser.add_argument('--model_file', "-m", type=str,
                        help='The model file. e.g. model_training.pt')
    parser.add_argument('-image', '-i', type=str,
                        help="image to be inferred")

    args = parser.parse_args()
    
    if args.model_file.split('.')[-1] == 'pt':
        torch_infer(args)
    else:
        onnx_infer(args)

def img_preprocess(img_path):
    img = Image.open(img_path)
    # 先变成opencv
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rows, cols = img.shape
    new_row = 32
    new_col = int(cols * new_row / rows)

    # 需要按row缩放到 32 x n
    img = cv2.resize(img, (new_col, new_row))

    # pad 到 32 x 384
    pad = 384 - new_col
    img = np.pad(img, ((0, 0), (0, pad)), 'constant', constant_values=(255, 255))

    # 颜色取反
    if True:
        img = 255 - img
    
    return img

def torch_infer(args):
    net = Model2(len(index_to_word), 1).to(device)
    model_file_path = args.model_file
    net.load_state_dict(torch.load(
            model_file_path, map_location=torch.device(device)))
    net.eval()

    img = img_preprocess(args.image)
    if False:
        cv2.imshow("img", img)
        cv2.waitKey()

    img = img / 255.0

    with torch.no_grad():
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        predict = predict_net(net, x)
        print(predict)
def onnx_infer(args):
    onnx_model = onnx.load(args.model_file)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(args.model_file)
    img = img_preprocess(args.image)
    # 增加batch维度
    img = img / 255.0
    img = img[np.newaxis, np.newaxis, :, :]
    img = img.astype(np.float32)
    
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)

    indices = np.argmax(ort_outs[0], axis=2)
    
    words = []
    for j in range(indices.shape[0]):
        word = index_to_word[int(indices[j, 0])]
        words.append(word)
    ans = arr_to_string(words)
    print(ans)


if __name__ == "__main__":
    main()