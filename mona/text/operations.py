import random


# 还有其他的提pr吧，先不找了
operations_names = [
    # 大世界
    "烹饪",
    "合成",
    "锻造",
    "坐下",
    "调查",
    "召唤浪船",
    "再试", # 清理背包再试的再试
    "激活",
    "启动",
    "向左",
    "向右",
    "阅读",
    "阅读「圣章石」",
    "接触征讨之花",
    "触碰元件",
    "挖掘",
    "拾取", # 旋曜玉帛
    "旋转",
    "触摸",
    "七天神像",
    "进入教堂",
    "开启挑战",
    # "开启试炼", 狼王的, 放秘境里了
    "接触剑柄",
    "参拜尊像",
    "召唤草种子",
    "召唤雷种子",
    "普通的宝箱",
    "精致的宝箱",
    "珍贵的宝箱",
    "华丽的宝箱",
    "奇馈宝箱",
    "顺时针旋转90度",
    "逆时针旋转90度",

    # 尘歌壶
    "进入「尘歌壶」",
    "更改洞天音乐",
    "进入宅邸",
    "邀请",
    "搭配「梦里花」",
    "种植于「楚此渚田」",
    "种植于「玄此玉田」",
    "种植于「薿此芝田」",
    "构筑「花庭」",
    "重置计分",
    "养鱼",
    "离开「尘歌壶」"

]

def random_operation_name():
    return random.choice(operations_names)