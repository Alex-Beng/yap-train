import random

# 所有可能交互的角色，包括NPC
characters_name = [
    "珐露珊",
    "流浪者",
    "纳西妲",
    "莱依拉",
    "赛诺",
    "坎蒂丝",
    "妮露",
    "柯莱",
    "多莉",
    "提纳里",
    "久岐忍",
    "鹿野院平藏",
    "夜兰",
    "瑶瑶",
    "神里绫人",
    "云堇",
    "八重神子",
    "申鹤",
    "荒泷一斗",
    "五郎",
    "托马",
    "埃洛伊",
    "珊瑚宫心海",
    "雷电将军",
    "九条裟罗",
    "宵宫",
    "早柚",
    "神里绫华",
    "枫原万叶",
    "优菈",
    "烟绯",
    "罗莎莉亚",
    "胡桃",
    "魈",
    "甘雨",
    "阿贝多",
    "钟离",
    "辛焱",
    "达达利亚",
    "迪奥娜",
    "可莉",
    "温迪",
    "刻晴",
    "莫娜",
    "七七",
    "迪卢克",
    "琴",
    "砂糖",
    "重云",
    "诺艾尔",
    "班尼特",
    "菲谢尔",
    "凝光",
    "行秋",
    "北斗",
    "香菱",
    "雷泽",
    "芭芭拉",
    "丽莎",
    "凯亚",
    "安柏",
    "白术",
    "卡维",
    "瑶瑶",
    "艾尔海森",
    "迪希雅",
    "米卡",
    "绮良良",
    # 4.0新角色
    "林尼",
    "琳妮特",
    "菲米尼",

    # 尘歌壶
    "派蒙",
    "阿圆",
    
    # 大世界，不想写爬虫了
    # 写点常见的好了
    "凯瑟琳",
    "莎拉",
    "优律",
    "玛乔丽",
    "瓦格纳",
    "昆恩",
    "芙罗拉",
    "提米",
    "蒂玛乌斯",
    "布兰琪",
    "艾琳",
    "维克多", 
    "德安公",
    "岚姐",
    "亨利莫顿",
    "斯万",
    "萨基",
    "齐格芙丽雅",
    "米哈伊尔",

    # 4.0 新NPC
    '爱贝儿',
    '莉雅丝',
    '维吉尔',
    '布蕾莘',
    '玛梅赫',
    '梅雅德',
    '加维',
    
    # 杂项
    "石碑",
    "无名学者的记事",
    "古旧的佣兵笔记·",
    "破损的日志",
    

    # 角色界面
    "元素充能效率",
    "生命值",
    "合成次数1",
    "暴击率",
    "攻击力",
    "暴击伤害",
]


def random_character_name():
    return random.choice(characters_name)
