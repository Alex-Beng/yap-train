import random
from .domains import domain_names

# UI 界面上的词
# 1. 类似pickup的需叠加黑底的
# 2. 直接与背景叠加的
# 3. 纯色背景，且是白色为底的
ui_names_pickup_like = [
    # for treasure killer
    "点击进入",

    # press Y
    "进入世界申请（",
    "秘境挑战组队邀请",
        
    # tp names
    # showing in list
    "七天神像-风",
    "七天神像-岩",
    "七天神像-雷",
    "七天神像-草",
    "七天神像-水",
    "七天神像-火",
    "七天神像-冰",

    "传送锚点",
    "口袋锚点",
    "外景锚点",
    "宅邸",

    "地脉衍出·启示之花",
    "地脉衍出·藏金之花",
    "标记",
    "世界任务",
    "晶蝶诱捕装置",
    "浪船",

    "冒险家协会",
    "炼金",    
    "铁匠铺",
    "蒙德钓鱼协会",
    "「西风骑士团」后勤",
    "天使的馈赠",
    "猎鹿人",
    "荣光之风",
    "猫尾酒馆",
    "西风骑士团",
    "忍冬之树",
    "西风大教堂",

    "冒险家协会",
    "炼金",    
    "铁匠铺",
    "璃月钓鱼协会",
    '「玩具摊」',
    "万民堂",
    "「璃月总务司」干事",
    "群玉阁",
    "南十字·死兆星",
    "明星斋",

    "冒险家协会",
    "炼金",
    "铁匠铺",
    "稻妻钓鱼协会",
    "鸣神大社·神樱",
    '「社奉行」吏僚',
    "神里屋敷",
    '「设计师」若紫',
    "木漏茶室",
    "志村屋",
    "根付之源",

    "冒险家协会",
    "炼金",
    "铁匠铺",
    "须弥钓鱼协会"
    '「教令院」联络官',
    "梅娜卡里商铺",
    "普斯帕咖啡馆",
    "兰巴德酒馆",
    "智慧宫",
    "净善宫",
    "梦之树",

    "冒险家协会",
    "炼金",
    # "铁匠铺",
    "博蒙特工坊", # 咋不叫铁匠铺呢
    "枫丹钓鱼协会",
    "德波大饭店",
    "露泽咖啡厅",
    "《蒸汽鸟报》主编",
    "沫芒宫",
    "白棠珍奇屋",
    "露景泉",
    "欧庇克莱歌剧院",
    '「公爵」办公室',
    "梅洛彼得堡",
    "福利餐",

    # 活动秘境，算了，没必要搞

    # TODO: domains
    # TODO: monsters
]
ui_names_raw_mix = [
    # click plot
    "自动",
    "播放中",

    # for treasure killer
    "岛上无贼",
]
ui_names_pure = [
    # press Space
    "Space",

    # tp click names on right bottom
    # click
    "传送",
    "传送至猫尾酒馆",
    "传送至蒙德城",
    "传送至覆雪之路",
    "传送至璃月港",
    "传送至稻妻城",
    "传送至鸣神岛",
    "传送至影向山",
    "传送至须弥城",
    "传送至桓那兰那",
    "传送至枫丹廷·利奥奈区",
    "传送至枫丹廷·沫芒宫",
    "传送至枫丹廷·纳博内区",
    "传送至枫丹廷·瓦萨里回廊",
    "传送至露景泉",
    "传送至欧庇克莱歌剧院",
    "传送至层岩巨渊·地下矿区",
    # not click
    "确认",
    "追踪",
    "停止追踪",
]

ui_names = ui_names_pickup_like + ui_names_raw_mix + ui_names_pure

# to set, for speeding up
ui_names_raw_mix, ui_names_pickup_like, ui_names_pure = set(ui_names_raw_mix), set(ui_names_pickup_like), set(ui_names_pure)

ui_names += domain_names
ui_names += [
    "无相之雷",
    "无相之风",
    "急冻树",
    "北风的王狼，奔狼的领主",
    "无相之岩",
    "纯水精灵",
    "爆炎树",
    "古岩龙蜥",
    "无相之冰",
    "魔偶剑鬼",
    "无相之火",
    "恒常机关阵列",
    "无相之水",
    "雷音权现",
    "黄金王兽",
    "深海龙蜥之群",
    "遗迹巨蛇",
    "掣电树",
    "翠翎恐蕈",
    "兆载永劫龙兽",
    "半永恒统辖矩阵",
    "无相之草",
    "风蚀沙虫",
    "神罪浸礼者",
    "「冰风组曲」",
    "铁甲熔火帝皇",
    "实验性场力发生装置",
    "千年珍珠骏麟",
    "水形幻人",
    
    "愚人众·萤术士",
    "深渊法师",
    "遗迹守卫",
    "丘丘暴徒",
    "愚人众·债务处理人",
    "遗迹猎者",
    "幼岩龙蜥",
    "丘丘王",
    "岩龙蜥",
    "愚人众·冬国仕女",
    "遗迹机兵",
    "兽境之狼",
    "深海龙蜥",
    "黑蛇众",
    "遗迹龙兽",
    "元能构装体",
    "圣骸兽",
    "丘丘游侠",
    "浊水幻灵",
    "隙境原体",
    "愚人众·役人",

    "愚人众·先遣队",
    "骗骗花",
    "丘丘人",
    "丘丘射手",
    "丘丘萨满",
    "史莱姆",
    "盗宝团",
    "野伏众",
    "飘浮灵",
    "镀金旅团",
    "蕈兽",
    "发条机关",
    "原海异种",
]

# 添加一个点击tp的白名单，然后点击最上面的
ui_names_click_to_tp = [
    "传送",
    "传送至猫尾酒馆",
    "传送至蒙德城",
    "传送至覆雪之路",
    "传送至璃月港",
    "传送至稻妻城",
    "传送至鸣神岛",
    "传送至影向山",
    "传送至须弥城",
    "传送至桓那兰那",
    "传送至枫丹廷·利奥奈区",
    "传送至枫丹廷·沫芒宫",
    "传送至枫丹廷·纳博内区",
    "传送至枫丹廷·瓦萨里回廊",
    "传送至露景泉",
    "传送至欧庇克莱歌剧院",
    "传送至层岩巨渊·地下矿区",

    "七天神像-风",
    "七天神像-岩",
    "七天神像-雷",
    "七天神像-草",
    "七天神像-水",
    "七天神像-火",
    "七天神像-冰",

    "传送锚点",
    "口袋锚点",
    "外景锚点",
    "宅邸",

    # TODO: domains
]
ui_names_click_to_tp += domain_names

def random_ui_name():
    rand_func = [
        lambda: random.choice(list(ui_names_pickup_like)),
        lambda: random.choice(list(ui_names_raw_mix)),
        lambda: random.choice(list(ui_names_pure)),
    ]
    weights = [
        # len(ui_names_pickup_like),
        # len(ui_names_raw_mix), 
        0,0,
        len(ui_names_pure),
    ]
    return random.choices(rand_func, weights=weights)[0]()