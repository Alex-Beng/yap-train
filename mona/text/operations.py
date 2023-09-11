import random


# 还有其他的提pr吧，先不找了
operations_names = [
    # 大世界
    "烹饪",
    "合成",
    "锻造",
    "坐下",
    "调查",
    "寻人启事",
    "召唤浪船",
    "驾驶浪船",
    "深赤之石",
    "参量质变仪",
    "晨曦酒庄招募书",
    "调查酒杯",
    "求职招聘互助板",
    "告示",
    "播撒",
    "钓鱼",
    # "再试", # 清理背包再试的再试
    "采摘", # 落叶归风
    '回收',
    '乱七八糟的脚印',
    "激活",
    "启动",
    "操作",
    "向左",
    "向右",
    "阅读",
    "观察",
    "交互",
    "解锁", # 键纹
    "标记「圣章石」",
    "领取奖励",
    "接触征讨之花",
    "接触地脉之花",
    "接触地脉溢口",
    "触碰元件",
    "挖掘",
    "拾取", # 旋曜玉帛 & 圣章石 & 紧急维修木板 & 书页
    "燃放烟花", # 烟花试玩计划 宵宫！！！
    "熔炼烟花",
    "燃放准备",
    "旋转",
    "触摸",
    "掩埋",
    "石碑",
    "公告板",
    "广告板",
    "观景点",
    "传送锚点",
    "进入酒馆",
    "进入教堂",
    "离开教堂",
    "猫尾酒馆海报",
    "开启",
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
    "点燃生之烛",
    "挑战「手鞠游戏」",
    "布置「手鞠」",
    "调小火力", # 全能美食队
    "调大火力",
    "抓捕",     # 猫之迹
    "开始拍照",  # 猫的留影
    "拿取物资", # 君子白日闯
    # tx7 377


    # 供奉
    "七天神像",
    "神樱",
    "忍冬之树",
    "梦之树",
    "甘露池",
    '露景泉',

    "进入木漏茶室",

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
    "离开「尘歌壶」",

    # 杂项
    # 奇奇怪怪的
    "无名学者的记事",
    "古旧的佣兵笔记·其二",
    "古旧的佣兵笔记·",
    "破损的日志",
    "业余炼金术士的日志",

    # 雪山
    "编号Hu-42318的记录",
    "编号Hu-96917的记录",
    "编号Hu-31122的记录",
    "编号GN-Hu-68513的记录",
    "编号Hu-16180的记录",
    "编号Hu-73011的记录",
    "编号Hu-81122的记录",
    "编号Hu-21030的记录",
    "编号Hu-57104的记录",

    # 4.0 操作
    "等待巡轨船…",
    '进入',
    '放置',
    '芒性能量块',
    '荒性能量块',
    '激活回响海螺',
    '呼叫升降机',
    "进入沫芒宫",
    '移动',
    "召回「安东·罗杰飞行器」",
    "召回「安东·罗杰飞行器",

]

def random_operation_name():
    return random.choice(operations_names)