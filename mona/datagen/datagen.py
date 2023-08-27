import random
import os

from PIL import Image, ImageFont, ImageDraw
import numpy as np

from mona.text import lexicon
from mona.text.artifact_name import random_monster_artifact_name, random_treasure_artifact_name, random_check_point_artifact_name
from mona.text.characters import random_character_name
from mona.text.domains import random_domain_name
from mona.text.material import random_material_name
from mona.text.operations import random_operation_name
from mona.text.weapons import random_weapon_name

from mona.config import config
from mona.datagen.pre_process import pre_process

# 4k分辨率最大对应84号字，900p分辨率最小对应18号字
# Yap需要固定字号
fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(96, 104)]




random_funcs = [
    random_monster_artifact_name,
    random_treasure_artifact_name,
    random_check_point_artifact_name,
    random_character_name,
    random_domain_name,
    random_material_name,
    random_operation_name,
    random_weapon_name,
    # hard code difficult name
    lambda : random.choice(['瑶瑶', '绮良良', 
                            '七七', '落落莓', 
                            '墩墩桃', '调查',
                            '泡泡桔', '嘟嘟莲',
                            '甜甜花', '钩钩果',
                            '松果', '松茸',
                            '棱镜', '棱晶',
                            '孢子', '种子',
                            '班尼特', '琳妮特',
                            "驾驶浪船",
                            "太山府",

                            '浊水的一','浊水的一','浊水的一',
                            '浊水的一掬', '浊水的一滴','浊水的一掬', '浊水的一滴','浊水的一掬', '浊水的一滴',
                            "地脉的旧枝", "地脉的枯叶", "地脉的新芽",
                            "旧枝", "枯叶", "新芽", "地脉的",
                            "开启试炼", "开启挑战",
                            '水晶蝶', '水晶块', '冰晶蝶', "晶蝶",
                            '兽肉', '鱼肉', 
                            "蓝角蜥", "红角蜥", "绿角蜥", '角蜥',
                            '隐兽指爪', '隐兽利爪','隐兽指爪', '隐兽利爪', '隐兽',
                            '荒性能量块', '芒性能量块',
                            "落日鳅鳅", "金鳅鳅", "晴天鳅鳅", '鳅鳅',
                            "藤纹陆鳗鳗", "深海鳗鳗", "赤鳍陆鳗鳗", "流沙鳗鳗", '鳗鳗',
                            '胡桃', '阅读',
                            '黑晶号角', '黑铜号角', '号角', '黑晶号', '黑铜号',
                            '召唤草种子', '召唤雷种子',
                            "晦暗刻像", "幽邃刻像", "夤夜刻像",'刻像',
                            '精锻用杂矿', '精锻用良矿', '精锻用魔矿', '精锻用', '魔矿', '良矿', '杂矿',
                            ]),
    # single word
    lambda : random.randint(0, 5) * (' ') + random.choice(lexicon),
    # twins word
    lambda : random.randint(0, 4) * (' ') + random.choice(lexicon) * 2,
    # 三字
    lambda : random.randint(0, 3) * (' ') + random.choice(lexicon) * 3,
    # 四字
    lambda : random.randint(0, 2) * (' ') + random.choice(lexicon) * 4,
    # dual word
    lambda : random.randint(0, 4) * (' ') + random.choice(lexicon) + random.choice(lexicon),
]
# 加大random_artifact_count的权重，因为连续数字识别是CRNN模型的难点，这对于副词条识别也有帮助。
random_weights = [
    3,
    1,
    1,
    1,
    1,
    5,
    3,
    1,
    1,
    # 0.6, 0.6, 0.6, 0.6, 0.6,
    8, 8, 8, 8, 8

]

random_funcs_genshin = random_funcs[:9]
random_weights_genshin = random_weights[:9]


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

# 仅生成原神中的names
def random_text_genshin_distribute():
    func = random.choices(
        population=random_funcs_genshin,
        weights=random_weights_genshin,
        k=1
    )
    return func[0]()


# 反射！反向控制！传入对象！JVAV！牛的不行！
def generate_image(rand_func=random_text):
    color1 = rand_color_1()
    color2 = rand_color_2()

    # 通过控制初始画布的宽度来指定 字符宽度缩放
    img = Image.new("RGB", (2000+ random.randint(-100, 100), 120), color1)
    # img = Image.new("RGB", (config["train_width"], config["height"]), color1)
    draw = ImageDraw.Draw(img)

    x = random.randint(10, 600)
    y = random.randint(-20, 30)
    # 模拟糟糕的阈值带来的粗笔画
    sk_w = random.randint(0, 2)
    text = rand_func()

    draw.text((x, y), text, color2, font=random.choice(fonts), stroke_width=sk_w)

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
import json
from PIL import Image

def js_dp(obj, path):
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

def js_ld(path):
    return json.load(open(path, 'r', encoding='utf-8'))

# 使用pickle读入预先存放的arrays，省去随机读取的时间
genshin_y_imgs = pickle.load(open('./genshin_y_imgs.pkl', 'rb'))
genshin_y = pickle.load(open('./genshin_y.pkl', 'rb'))

assert(len(genshin_y_imgs) == len(genshin_y))
genshin_n = len(genshin_y)

'''
genshin_x = js_ld('../yap/xx.json')
genshin_y = js_ld('../yap/yy.json')

# 真实标签仅使用空白数据，无需验证lexicon
def text_all_in_lexicon(text):
    for c in text:
        if c not in lexicon:
            return False
    return True

genshin_x = [genshin_x[i] for i in range(len(genshin_x)) if text_all_in_lexicon(genshin_y[i])]
genshin_y = [genshin_y[i] for i in range(len(genshin_y)) if text_all_in_lexicon(genshin_y[i])]

# only save the empty label xys
# genshin_x = [genshin_x[i] for i in range(len(genshin_x)) if genshin_y[i] == '']
# genshin_y = [genshin_y[i] for i in range(len(genshin_y)) if genshin_y[i] == '']

assert(len(genshin_x) == len(genshin_y))
print(f'genshin data len: {len(genshin_x)}')
root_path = "../yap/"
genshin_n = len(genshin_x)
# for speed up
# 预读入加速训练 吞吐: 500->550
genshin_y_imgs = []
for i in range(genshin_n):
    path = os.path.join(root_path, genshin_x[i])
    with Image.open(path) as img:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        genshin_y_imgs.append(img)
'''

def generate_mix_image(rand_func=random_text, ratio=0.5):

    # 一半真实数据，一半生成数据
    # 真实数据仅用空白数据
    if random.random() < ratio:
        idx = random.randint(0, genshin_n - 1)
        text = genshin_y[idx]
        # while not text_all_in_lexicon(text):
        #     print(f"[warning] {text} not in lexicon")
        #     idx = random.randint(0, genshin_n - 1)
        #     text = genshin_y[idx]

        # path = os.path.join(root_path, genshin_x[idx])
        
        img = genshin_y_imgs[idx]
        # img = Image.open(path)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # img = cv2.resize(img, (145, 32))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        img = Image.fromarray(img)
        return img, text
        

        # if text == '':
        #     img_way = random.randint(1, 5)
        #     if img_way == 1:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=0)
        #     elif img_way == 2:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=255)
        #     elif img_way == 3:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_DEFAULT, value=0)
        #     elif img_way == 4:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_REFLECT, value=0)
        #     elif img_way == 5:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_REPLICATE, value=0)    
            
        #     img = Image.fromarray(img)
        #     return img, text
        # else:
        #     img_way = random.randint(1, 2)
        #     if img_way == 1:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=0)
        #     elif img_way == 2:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-145, cv2.BORDER_CONSTANT, value=255)
            
        #     img = Image.fromarray(img)
        #     return img, text
        
    else:
        return generate_image(rand_func)
    

# Generate and return sample before/after pre_process
# 已弃用
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
