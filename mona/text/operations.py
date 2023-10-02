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
    "浪船锚点",
    "深赤之石",
    "参量质变仪",
    "晨曦酒庄招募书",
    "调查酒杯",
    "求职招聘互助板",
    "告示",
    "告示板",

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
    "旋转元件",
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
    "猫尾酒馆海报",
    "猫尾酒馆留言板",
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
    "「御神签箱」",
    "开门",
    "图书馆使用规范",
    "调小火力", # 全能美食队
    "调大火力",
    "抓捕",     # 猫之迹
    "开始拍照",  # 猫的留影
    "拿取物资", # 君子白日闯
    "上行", # 智慧宫电梯
    "下行",
    "谁人的题诗",
    "珠钿舫鉴珍录",
    "万民堂海报",
    "天平" # 群玉阁
    "整齐堆放的卷轴",
    "群玉阁情报墙",
    "广告牌", # 大巴扎
    "箴言栏",
    "「木南料亭」新菜告知",
    "告示牌",
    "通知！",
    "遗落的文件",
    "充能",
    "神秘的纪事",
    "古老的工作日志",
    "翻转", # 沙漠沙漏
    "深境螺旋",
    "改变频率",
    "退出秘境",
    "关闭",


    "前往顶层",
    "前往中层",
    "前往底层",

    # 供奉
    "七天神像",
    "神樱",
    "忍冬之树",
    "梦之树",
    "甘露池",
    '露景泉',

    # 进入离开
    '进入',
    "进入「尘歌壶」",
    "进入宅邸",
    "进入酒馆", # 蒙德
    "进入教堂",
    "进入猫尾酒馆",
    "进入骑士团",
    "进入群玉阁", # 璃月
    "进入琉璃亭",
    "进入新月轩",
    "进入北国银行",
    "进入木漏茶室", # 稻妻
    "进入乌有亭", 
    "进入净善宫", # 须弥
    "进入智慧宫",
    "进入咖啡馆",
    "进入「艾尔海森的住宅」",
    "进入德波大饭店", # 枫丹
    "进入沫芒宫",
    "进入歌剧院",
    "离开",
    "离开「尘歌壶」",
    "离开宅邸",
    "离开酒馆", # 蒙德
    "离开教堂",
    "离开猫尾酒馆",
    "离开骑士团",
    "离开群玉阁", # 璃月
    "离开琉璃亭",
    "离开新月轩",
    "离开北国银行",
    "离开木漏茶室", # 稻妻
    "离开乌有亭",
    "离开净善宫", # 须弥
    "离开智慧宫",
    "离开咖啡馆",
    "离开「艾尔海森的住宅」",
    "离开德波大饭店", # 枫丹
    "离开沫芒宫",
    "离开歌剧院",

    # 尘歌壶
    "更改洞天音乐",
    "邀请",
    "搭配「梦里花」",
    "种植于「楚此渚田」",
    "种植于「玄此玉田」",
    "种植于「薿此芝田」",
    "构筑「花庭」",
    "重置计分",
    "养鱼",

    # 杂项
    # 奇奇怪怪的
    "无名学者的记事",
    "古旧的佣兵笔记·其二",
    "古旧的佣兵笔记·",
    "破损的日志",
    "业余炼金术士的日志",
    "破破烂烂的笔记",
    "遗落的盗宝日记",
    "神秘的书页",   # 鹮巷物语

    # 雪山
    "编号Hu-42318的记录",
    "编号Hu-96917的记录",
    "编号Hu-31122的记录",
    "编号GN/Hu-68513的记录",
    "编号Hu-16180的记录",
    "编号Hu-73011的记录",
    "编号Hu-81122的记录",
    "编号Hu-21030的记录",
    "编号Hu-57104的记录",

    # 渊下宫
    "转换至「白夜」",
    "转换至「常夜」",

    # 4.0 操作
    "等待巡轨船…",
    '放置',
    '芒性能量块',
    '荒性能量块',
    '激活回响海螺',
    '呼叫升降机',
    "前往海露港下层",
    "前往海露港上层·通往枫丹",
    "四层「沫芒宫」",
    "三层「娜维娅线」",
    "二层「克莱门汀线」",
    "一层「总站大厅」",
    '移动',
    "召回「安东·罗杰飞行器」",
    # "召回「安东·罗杰飞行器", # 没有必要缩短了
    "安置水之核",
    "古老的调查报告",
    "古老的日志",

]

def random_operation_name():
    return random.choice(operations_names)