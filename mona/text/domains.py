import random


# 各种挑战副本
domain_names = [
    
    # 圣遗物秘境
    "仲夏庭园",
    "无妄引咎密宫",
    "华池岩岫",
    "铭记之谷",
    "孤云凌霄之处",
    "芬德尼尔之顶",
    "山脊守望",
    "椛染之庭",
    "沉眠之庭",
    "岩中幽谷",
    "缘觉塔",
    "赤金的城墟",
    "熔铁的孤塞",

    # 天赋秘境
    "忘却之峡",
    "太山府",
    "菫色之庭",
    "昏识塔",

    # 武器秘境
    "塞西莉亚苗圃",
    "震雷连山密宫",
    "砂流之庭",
    "有顶塔",

    # 周本秘境
    "开启试炼",
    "深入风龙废墟",
    "进入「黄金屋」",
    "「伏龙树」之底",
    "鸣神岛·天守",
    "梦想乐土之殁",
    "净琉璃工坊",
    "肇始之乡",

    # 一次性秘境
    "西风之鹰的庙宇",
    "北风之狼的庙宇",
    "南风之狮的庙宇",
    "鹰之门",

    "墟散人离之处",
    "华清归藏密宫",
    "曲径通幽之处",

    # 3.0
    "伞盖的荫蔽",
    "童梦的切片",
    "河谷的黯道",
    "晴雨的经纬",
    # 3.1
    "千柱的花园",
    "蜃景祭场",
    "赤沙之槛",
    # 3.4
    "五绿洲之殿堂",
    "亡者之城",
    # 3.6
    "地中的香海",
    "净罪之井",
]

def random_domain_name():
    return random.choice(domain_names)