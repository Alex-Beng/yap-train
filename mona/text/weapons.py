import random

# 大世界掉落的武器
weapons_name = [
    "无锋剑",
    "银剑",

    "训练大剑",
    "佣兵重剑",

    "猎弓",
    "历练的猎弓",

    "学徒笔记",
    "口袋魔法书",

    "新手长枪",
    "铁尖枪",

    # 三星 宝箱获得
    "吃虎鱼刀",
    "旅行剑",
    "白铁大剑",
    "飞天大御剑",
    "信使",
    "反曲弓",
    "异世界行记",
    "甲级宝珏",
    "钺矛",
    "白缨枪"
]

def random_weapon_name():
    return random.choice(weapons_name)