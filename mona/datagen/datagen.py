import random
import os

from PIL import Image, ImageFont, ImageDraw
import numpy as np

from mona.text import lexicon
from mona.text.artifact_name import random_artifact_name
from mona.text.stat import random_sub_stat, random_main_stat_name, random_main_stat_value
from mona.text.characters import random_equip
from mona.text.material import random_material_name, random_cant_hold_material
from mona.config import config
from mona.datagen.pre_process import pre_process

# 4k分辨率最大对应84号字，900p分辨率最小对应18号字
# 这里需要固定字号
fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(96, 104)]


def random_level():
    return "+" + str(random.randint(0, 20))


def random_artifact_count():
    # Random here, for online learning
    flag_ac = random.randint(0, 1500)
    return f"圣遗物 {flag_ac}/1500"


def random_number():
    n = random.randint(0, 1000000000)
    return f"{n}"



random_funcs = [random_artifact_name, random_material_name]
# 加大random_artifact_count的权重，因为连续数字识别是CRNN模型的难点，这对于副词条识别也有帮助。
random_weights = [0.3, 0.7]


def rand_color_1():
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)

    # temp = random.choice(background_colors)
    # r = min(255, temp[0] + random.randint(-20, 20))
    # g = min(255, temp[1] + random.randint(-20, 20))
    # b = min(255, temp[2] + random.randint(-20, 20))

    return r, g, b


def rand_color_2():
    r = random.randint(0, 100)
    g = random.randint(0, 100)
    b = random.randint(0, 100)

    # temp = random.choice(font_colors)
    # r = min(255, temp[0] + random.randint(-20, 20))
    # g = min(255, temp[1] + random.randint(-20, 20))
    # b = min(255, temp[2] + random.randint(-20, 20))

    return r, g, b


def random_text():
    func = random.choices(
        population=random_funcs,
        weights=random_weights,
        k=1
    )
    return func[0]()
    # return random_artifact_count()


def generate_image():
    color1 = rand_color_1()
    color2 = rand_color_2()

    # 通过控制初始画布的宽度来指定 字符宽度缩放
    img = Image.new("RGB", (2000+ random.randint(-100, 100), 120), color1)
    # img = Image.new("RGB", (config["train_width"], config["height"]), color1)
    draw = ImageDraw.Draw(img)

    x = random.randint(10, 20)
    y = random.randint(-20, 30)

    text = random_text()
    text = "烹饪"

    draw.text((x, y), text, color2, font=random.choice(fonts))

    # 使用大津法阈值
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (384, 32))
    cv2.threshold(img, 0, 255, cv2.THRESH_OTSU, img)
    img = cv2.bitwise_not(img)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # print(img.shape)
    # exit()
    img = Image.fromarray(img)

    return img, text

import pickle
import cv2
from PIL import Image

genshin_x = pickle.load(open("../yap/x.pk", "rb"))
genshin_y = pickle.load(open("../yap/y.pk", "rb"))
assert(len(genshin_x) == len(genshin_y))
print(f'genshin data len: {len(genshin_x)}')
root_path = "../yap/dumps/"
genshin_n = len(genshin_x)

def text_all_in_lexicon(text):
    for c in text:
        if c not in lexicon:
            return False
    return True

def generate_mix_image():

    # 一半真实数据，一半生成数据
    if random.random() < 0.5:        
        idx = random.randint(0, genshin_n - 1)
        text = genshin_y[idx]
        while not text_all_in_lexicon(text):
            idx = random.randint(0, genshin_n - 1)
            text = genshin_y[idx]
                
        img = cv2.imread(f'{root_path}{genshin_x[idx]}_raw.jpg')
        img = cv2.resize(img, (145, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]

        

        if text == '':
            img_way = random.randint(1, 5)
            if img_way == 1:
                img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=0)
            elif img_way == 2:
                img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=255)
            elif img_way == 3:
                img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_DEFAULT, value=0)
            elif img_way == 4:
                img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_REFLECT, value=0)
            elif img_way == 5:
                img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_REPLICATE, value=0)    
            
            img = Image.fromarray(img)
            return img, text
        else:
            img_way = random.randint(1, 2)
            if img_way == 1:
                img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=0)
            elif img_way == 2:
                img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=255)
            
            img = Image.fromarray(img)
            return img, text
        
    else:
        return generate_image()
    

# Generate and return sample before/after pre_process
def generate_image_sample():
    color1 = rand_color_1()
    color2 = rand_color_2()

    img = Image.new("RGB", (1200, 120), color1)
    draw = ImageDraw.Draw(img)
    x = random.randint(0, 20)
    y = random.randint(0, 5)

    text = random_text()
    # draw.text((x, y), text, color2, font=random.choice(fonts))

    # This would disable anit-aliasing
    # draw.fontmode = "1"

    # draw.text((20, 5), "虺雷之姿", color2, font=ImageFont.truetype("./assets/genshin.ttf", 80))
    draw.text((20, 5), "归风佳酿节节庆热度", color2, font=ImageFont.truetype("./assets/genshin.ttf", 80))

    img_processed = pre_process(img)
    return img, img_processed   
