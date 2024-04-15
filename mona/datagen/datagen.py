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
from mona.text.server_leak_names import random_server_leak_name
from mona.text.book_names import random_book_name
from mona.text.common_Chinese import random_chinese
from mona.text.field_operations import random_field_operation_name
from mona.text.ui_names import random_ui_name

from mona.text.artifact_name import monster_artifact_name, treasure_artifact_names, check_point_artifact_names
from mona.text.characters import characters_name
from mona.text.domains import domain_names
from mona.text.material import material_names
from mona.text.operations import operations_names
from mona.text.weapons import weapons_name
from mona.text.server_leak_names import server_leak_names
from mona.text.book_names import book_names
from mona.text.field_operations import field_operations_names
from mona.text.ui_names import ui_names, ui_names_raw_mix, ui_names_pickup_like, ui_names_pure


from mona.config import config
from mona.datagen.pre_process import pre_process

# 4k分辨率最大对应84号字，900p分辨率最小对应18号字
# Yap需要固定字号
fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(85, 104)]

def load_bg_imgs():
    path = "../yap/dumps_full_mona2/"
    # check path exists
    if not os.path.exists(path):
        return []
    # 获取文件夹下所有图片
    files = os.listdir(path)
    # 读取图片
    imgs = []
    for file in files:
        imgs.append(cv2.imread(path + file))
    return imgs

random_funcs = [
    random_monster_artifact_name,
    random_treasure_artifact_name,
    random_check_point_artifact_name,
    random_character_name,
    random_domain_name,
    random_material_name,
    random_operation_name,
    random_weapon_name,
    random_server_leak_name,
    random_book_name,
    random_field_operation_name,

    # hard code difficult name and new word
    lambda : random.choice([
        # "临瀑之城",
        # "加入奶泡",
        # "《遗珑埠不容错过的三种小吃》",
        # "决定上午的安排",
        "清水玉",
        
        "羽状鳍翅",
        "月色鳍翅",
        "渊光鳍翅",

        # "",
        # "阿蕾奇诺",
        # "突入邪恶巢穴","突入邪恶巢穴","突入邪恶巢穴",
        # "勇闯水妖王国","勇闯水妖王国","勇闯水妖王国",
        # "奏响回响海螺","奏响回响海螺","奏响回响海螺",
        # "主板调试","主板调试","主板调试",
        # "加入奶油",
        # "加入牛奶",
        # "阿嘟",

        # '进入世界申请（',
        # "秘境挑战组队邀请",
        # "红莲蛾",
        # "碎星铁矿", "观察手鞠",
        "自动",
        "播放中",
        # "安装键帽","安装键帽","安装键帽","安装键帽","安装键帽",
        # "海贼的日志",
        # "雨","晴","流转",
        # "异界余影",
        # '塔米米',
        # '字迹歪歪扭扭的记事',
        # '观察蒲公英',
        # '特别枫达专卖机',
        '思思','璐璐',
        '小麦', '小姜','小蒙',
        # '混沌容器','混沌装置', '混沌机关', '沉重号角', 
        # '混沌模块','混沌回路', '混沌枢纽', '黑铜号角', 
        # '混沌锚栓','混沌炉心', '混沌真眼', '黑晶号角', 

        # "老旧的役人怀表",
        # "役人的制式怀表",
        # "役人的时时刻刻",
            
        # "湖光铃兰",
        # "初露之源",
        # "无光丝线",
        # "无光涡眼",
        # "无光质块",

        # '墩墩桃', '调查', '薇塔',
        # '泡泡桔', '嘟嘟莲',
        # '甜甜花', '钩钩果',
        # '松果', '松茸',
        # '班尼特', '琳妮特',
        
        '水晶蝶', '水晶块', '冰晶蝶', 
        # "落日鳅鳅", "金鳅鳅", "晴天鳅鳅",
        # "藤纹陆鳗鳗", "深海鳗鳗", "赤鳍陆鳗鳗", "流沙鳗鳗", 
        ]),
]
random_weights = [
    len(monster_artifact_name) * 2,
    len(treasure_artifact_names) * 2,
    len(check_point_artifact_names) * 2,
    len(characters_name),
    len(domain_names),
    len(material_names) * 2,
    len(operations_names),
    len(weapons_name),
    len(server_leak_names),
    len(book_names),
    len(field_operations_names),

    400,
]
random_funcs_genshin = random_funcs.copy()
random_weights_genshin = random_weights.copy()

# norm the weights and print it
sum_weights = sum(random_weights)
random_weights = [w / sum_weights for w in random_weights]

# print([w / sum_weights for w in random_weights])
for w in random_weights:
    print(f"{w*100:.4f}", end=" ")
print()

random_funcs += [
    # random chinese
    lambda : random_chinese(1),
    lambda : random_chinese(2),
    lambda : random_chinese(3),
    lambda : random_chinese(4),
    lambda : random_chinese(5),
    lambda : random_chinese(6),
    lambda : random_chinese(7),
    lambda : random_chinese(8),

]
random_weights += [84, 3903, 2934, 2079, 1266, 477, 615, 246] # sample from genshin data



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
def generate_pure_bg_image(rand_func=random_text):
    color1 = rand_color_1()
    color2 = rand_color_2()

    # 通过控制初始画布的宽度来指定 字符宽度缩放
    img = Image.new("RGB", (2000 + random.randint(-80, 80), 120), color1)
    # img = Image.new("RGB", (config["train_width"], config["height"]), color1)
    draw = ImageDraw.Draw(img)

    x = random.randint(10, 120)
    y = random.randint(-12, 20)
    # 模拟糟糕的阈值带来的粗笔画
    sk_w = random.randint(0, 1)
    text = rand_func()
    # text = "冒险家罗尔德的日志·绝云间·奥藏天池"
    # text = "?？?？"
    rd_num = random.random()
    # if rd_num < 0.05:
    #     text = "冒险家罗尔德的日志·绝云间·奥藏天池"
    
    # if rd_num < 0.3 and len(text) > 3:
    #     text = text[:-1] if text != '' else text
    # elif rd_num > 0.7 and len(text) > 3:
    #     text = text[1:] if text != '' else text
    # if random.random() < 0.3:
    #     text = text[::-1]

    draw.text((x, y), text, color2, font=random.choice(fonts), stroke_width=sk_w)

    rd_num = random.random()
    if rd_num < 0:
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (384, 32))
        img = Image.fromarray(img)
        return img, text

    # 使用大津法阈值
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (384, 32))
    cv2.threshold(img, 0, 255, cv2.THRESH_OTSU, img)

    # 计算文字区域的最右边的白色像素所在的col
    # 用于后续的叠加
    img_right_white_col = 0
    for col in range(383, 0, -1):
        if img[:, col].max() == 255:
            img_right_white_col = col
            break

    img = cv2.bitwise_not(img)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # print(img.shape)
    # exit()
    # img = Image.fromarray(img)
    # return img, text

    # pipeline
    # 扣出文字
    # 随机选取32x384的背景
    # 背景叠加一个从左到右渐变黑的图片
    # 背景叠加文字

    img_inv = cv2.bitwise_not(img)
    
    bg_img = random.choice(bg_imgs)
    bg_r, bg_c, _ = bg_img.shape
    res_w, res_h = 384, 32

    x = np.random.randint(0, bg_c-res_w)
    y = np.random.randint(0, bg_r-res_h)

    res_img = bg_img[y:y+res_h, x:x+res_w].copy()
    # cv2.imshow("bg", res_img)
    
    # 随机选取的背景图
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)

    # 叠加渐变图
    black2white = np.full((32, 384), 0, dtype=np.uint8)
    white_thre = random.randint(180, 230)
    # white_thre = 180
    for i in range(384):
        pixel = i * 0.5
        black2white[:, i] = pixel
        if pixel > white_thre:
            black2white[:, i] = white_thre
    # 以比例混合
    cv2.addWeighted(black2white, 0.2, res_img, 0.8, 0, res_img)


    min_count_val = random.randint(white_thre//2+100, 255)
    # min_count_val = white_thre//2+100

    rand_img = np.full((32, 384), min_count_val, dtype=np.uint8)
    # img = cv2.addWeighted(rand_img, 0.2, img, 0.8, 0, img)
    # 将img中的白色像素点的值变为rand_img中的值，使用opencv的bitwise_and
    img = cv2.bitwise_and(rand_img, rand_img, mask=img)

    # 随机权重叠加字和背景
    wdg = random.uniform(0.4, 0.7)
    # wdg = 0.4
    res_img = cv2.addWeighted(res_img, wdg, img, 1-wdg, 0, res_img)

    # res_img 随机乘以一个系数
    max_pixel = res_img.max()
    max_ratio = 255 / max_pixel
    res_img = res_img * random.uniform(0.7, max_ratio)
    res_img = res_img.astype(np.uint8)

    res_img = Image.fromarray(res_img)
    return res_img, text

import lzma
import pickle
import cv2
import json
from PIL import Image

def js_dp(obj, path):
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

def js_ld(path):
    return json.load(open(path, 'r', encoding='utf-8'))

# 使用pickle读入预先存放的arrays，省去随机读取的时间
# TODO: remove this tryd
try:
    genshin_x_imgs = pickle.load(lzma.open('/media/alex/Data/genshin_x_imgs.pkl', 'rb'))
    genshin_y = pickle.load(lzma.open('/media/alex/Data/genshin_y.pkl', 'rb'))
except:
    backup_path = "D:/"
    backup_x_path = os.path.join(backup_path, 'genshin_x_imgs.pkl')
    backup_y_path = os.path.join(backup_path, 'genshin_y.pkl')
    if os.path.exists(backup_x_path) and os.path.exists(backup_y_path):
        genshin_x_imgs = pickle.load(lzma.open(backup_x_path, 'rb'))
        genshin_y = pickle.load(lzma.open(backup_y_path, 'rb'))
    else:
        genshin_x_imgs = []
        genshin_y = []


assert(len(genshin_x_imgs) == len(genshin_y))
genshin_n = len(genshin_y)
bg_imgs = load_bg_imgs()

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
genshin_x_imgs = []
for i in range(genshin_n):
    path = os.path.join(root_path, genshin_x[i])
    with Image.open(path) as img:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        genshin_x_imgs.append(img)
'''

def generate_pickup_image(rand_func=random_text, ratio=0.5):

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
        
        img = genshin_x_imgs[idx]
        # img = Image.open(path)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # img = cv2.resize(img, (221, 32))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        img = Image.fromarray(img)
        return img, text
        

        # if text == '':
        #     img_way = random.randint(1, 5)
        #     if img_way == 1:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-221, cv2.BORDER_CONSTANT, value=0)
        #     elif img_way == 2:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-221, cv2.BORDER_CONSTANT, value=255)
        #     elif img_way == 3:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-221, cv2.BORDER_DEFAULT, value=0)
        #     elif img_way == 4:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-221, cv2.BORDER_REFLECT, value=0)
        #     elif img_way == 5:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-221, cv2.BORDER_REPLICATE, value=0)    
            
        #     img = Image.fromarray(img)
        #     return img, text
        # else:
        #     img_way = random.randint(1, 2)
        #     if img_way == 1:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-221, cv2.BORDER_CONSTANT, value=0)
        #     elif img_way == 2:
        #         img = cv2.copyMakeBorder(img, 0,0,0,384-221, cv2.BORDER_CONSTANT, value=255)
            
        #     img = Image.fromarray(img)
        #     return img, text
        
    else:
        return generate_pure_bg_image(rand_func)


# 纯色比例 -> image & text
def generate_ui_image():
    color1 = rand_color_1()
    color2 = rand_color_2()

    # 通过控制初始画布的宽度来指定 字符宽度缩放
    img = Image.new("RGB", (2000 + random.randint(-80, 80), 120), color1)
    # img = Image.new("RGB", (config["train_width"], config["height"]), color1)
    draw = ImageDraw.Draw(img)

    x = random.randint(10, 120)
    y = random.randint(-12, 20)
    # 模拟糟糕的阈值带来的粗笔画
    sk_w = random.randint(0, 1)
    text = random_ui_name()
    
    draw.text((x, y), text, color2, font=random.choice(fonts), stroke_width=sk_w)

    if text in ui_names_pure:
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (384, 32))
        # random 255-pixel
        if random.random() < 0.5:
            img = cv2.bitwise_not(img)
        img = Image.fromarray(img)
        return img, text

    # 使用大津法阈值
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (384, 32))
    cv2.threshold(img, 0, 255, cv2.THRESH_OTSU, img)

    img = cv2.bitwise_not(img)

    # pipeline
    # 扣出文字
    # 随机选取32x384的背景
    # 背景叠加一个从左到右渐变黑的图片
    # 背景叠加文字
    
    bg_img = random.choice(bg_imgs)
    bg_r, bg_c, _ = bg_img.shape
    res_w, res_h = 384, 32

    x = np.random.randint(0, bg_c-res_w)
    y = np.random.randint(0, bg_r-res_h)

    res_img = bg_img[y:y+res_h, x:x+res_w].copy()
    
    # 随机选取的背景图
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)

    # 叠加渐变图
    black2white = np.full((32, 384), 0, dtype=np.uint8)
    if text in ui_names_pickup_like:
        white_thre = 384
        # white_thre = 180
        for i in range(384):
            pixel = i * 0.3
            black2white[:, i] = pixel
            if pixel > white_thre:
                black2white[:, i] = white_thre
        # 以比例混合
        cv2.addWeighted(black2white, 0.5, res_img, 0.5, 0, res_img)

        min_count_val = random.randint(100, 255)
        # min_count_val = white_thre//2+100

        rand_img = np.full((32, 384), min_count_val, dtype=np.uint8)
        # img = cv2.addWeighted(rand_img, 0.2, img, 0.8, 0, img)
        # 将img中的白色像素点的值变为rand_img中的值，使用opencv的bitwise_and
        img = cv2.bitwise_and(rand_img, rand_img, mask=img)

        # 随机权重叠加字和背景
        wdg = random.uniform(0.1, 0.5)
        # wdg = 0.4
        res_img = cv2.addWeighted(res_img, wdg, img, 1-wdg, 0, res_img)

        # res_img 随机乘以一个系数
        max_pixel = res_img.max()
        max_ratio = 255 / max_pixel
        res_img = res_img * random.uniform(0.7, max_ratio)

        res_img = res_img.astype(np.uint8)

        res_img = Image.fromarray(res_img)
        return res_img, text
    else: # in ui_names_raw_mix
        min_count_val = random.randint(50, 255)
        rand_img = np.full((32, 384), min_count_val, dtype=np.uint8)
        # img = cv2.addWeighted(rand_img, 0.2, img, 0.8, 0, img)
        # 将img中的白色像素点的值变为rand_img中的值，使用opencv的bitwise_and
        img = cv2.bitwise_and(rand_img, rand_img, mask=img)

        wdg = random.uniform(0.1, 0.3)
        res_img = cv2.addWeighted(res_img, wdg, img, 1-wdg, 0, res_img)

        res_img = res_img.astype(np.uint8)

        res_img = Image.fromarray(res_img)
        return res_img, text


# mix pickup and ui
def generate_mix_image(pickup_rand_func=random_text, pickup_genshin_ratio=0.5, pickup_ratio=0.5):
    if random.random() < pickup_ratio:
        return generate_pickup_image(pickup_rand_func, pickup_genshin_ratio)
    else:
        return generate_ui_image()


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
    draw.text((20, 5), "冒险家罗尔德的日志·绝云间·奥藏天池", color2, font=ImageFont.truetype("./assets/genshin.ttf", 80))

    img_processed = pre_process(img)
    return img, img_processed   
