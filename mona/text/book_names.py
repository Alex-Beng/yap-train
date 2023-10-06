import random

book_names = [
    "亡国的美奈姬·卷五",
    # "神秘的书页" # in ops
    "少女薇拉的忧郁·卷一",
    "少女薇拉的忧郁·卷二",
    "少女薇拉的忧郁·卷三",
    "少女薇拉的忧郁·卷四",
    "少女薇拉的忧郁·卷五",
    "少女薇拉的忧郁·卷六",
    "少女薇拉的忧郁·卷七",
    "少女薇拉的忧郁·卷八",
    "少女薇拉的忧郁·卷九",
    "少女薇拉的忧郁·卷十",
    "希鲁伊与希琳的故事·卷一",
    "希鲁伊与希琳的故事·卷二",
    "怪盗与名侦探：虹彩胸针之谜·卷一",
    "新六狐传·三",
    "枫丹动物寓言集·卷二",
    "枫丹动物寓言集·卷三",
    "沉秋拾剑录·五",
    "浮槃歌卷·卷一",
    "浮槃歌卷·卷二",
    "浮槃歌卷·卷三",
    "牧童与魔瓶的故事",
    "犬又二分之一·一",
    "犬又二分之一·二",
    "犬又二分之一·三",
    "犬又二分之一·四",
    "犬又二分之一·五",
    "犬又二分之一·六",
    "犬又二分之一·七",
    "犬又二分之一·八",
    "犬又二分之一·九",
    "碎梦奇珍·月光",
    "碎梦奇珍·琉璃",
    "碎梦奇珍·石心",
    "神霄折戟录·第二卷",
    "神霄折戟录·第三卷",
    "神霄折戟录·第六卷",
    "荒山孤剑录·一",
    "荒山孤剑录·二",
    "荒山孤剑录·三",
    "蒲公英海的狐狸·卷二",
    "蒲公英海的狐狸·卷三",
    "蒲公英海的狐狸·卷四",
    "蒲公英海的狐狸·卷五",
    "蒲公英海的狐狸·卷六",
    "蒲公英海的狐狸·卷七",
    "蒲公英海的狐狸·卷八",
    "蒲公英海的狐狸·卷九",
    "蒲公英海的狐狸·卷十",
    "蒲公英海的狐狸·卷十一",
    "遐叶论经·卷一",
    "遐叶论经·卷二",
    "遐叶论经·卷三",
    "野猪公主·卷一",
    "野猪公主·卷二",
    "野猪公主·卷三",
    "野猪公主·卷四",
    "野猪公主·卷五",
    "野猪公主·卷六",
    "阿赫玛尔的故事",
    "雷穆利亚衰亡史·卷二",
    "「东王」史辩",
    "丘丘人习俗考察·卷一",
    "丘丘人习俗考察·卷二",
    "丘丘人习俗考察·卷三",
    "丘丘人习俗考察·卷四",
    "丘丘人诗歌选·上卷",
    "丘丘人诗歌选·下卷",
    "侍从骑士之歌·上篇",
    "侍从骑士之歌·下篇",
    "侠客记·山叟篇",
    "侠客记·留尘",
    "冒险家罗尔德的日志·地中之盐",
    "冒险家罗尔德的日志·轻策山庄",
    "冒险家罗尔德的日志·绝云间·奥藏天池",
    "冒险家罗尔德的日志·渌华池",
    "冒险家罗尔德的日志·瑶光滩",
    "冒险家罗尔德的日志·孤云阁",
    "冒险家罗尔德的日志·绝云间·庆云顶",
    "冒险家罗尔德的日志·青墟浦",
    "冒险家罗尔德的日志·龙脊雪山",
    "冒险家罗尔德的日志·离岛",
    "冒险家罗尔德的日志·鹤观",
    "巫女曚云小传",
    "帝君尘游记·一",
    "帝君尘游记·二",
    "帝君尘游记·三",
    "帝君尘游记·四",
    "林间风·故事拔萃节选",
    "林间风·龙之书",
    "浮浪记·潮起",
    "浮浪记·狂涛",
    "清泉之心·一",
    "清泉之心·二",
    "清泉之心·三",
    "清泉之心·四",
    "温妮莎传奇·上篇",
    "温妮莎传奇·下篇",
    "珊瑚宫民间信仰初勘",
    "珊瑚宫记",
    "璃月风土志·绣球",
    "璃月风土志·迎神",
    "日月前事",
    "白夜国地理水文考",
    "深海龙蜥实验记录",
    "光昼影底集",
    "竹林月夜·一",
    "竹林月夜·四",
    "绝云记闻·无妄",
    "给东东的信",
    "谁人的日志·其一·瑶光滩",
    "谁人的日志·其二·归离原",
    "谁人的日志·其三·绝云间",
    "谁人的日志·其四·璃月港",
    "谁人的日志·其五·刃连岛",
    "连心珠·卷一",
    "连心珠·卷四",
    "连心珠·卷五",
    "醉客轶事·第一卷",
    "醉客轶事·第二卷",
    "醉客轶事·第三卷",
    "醉客轶事·第四卷",
    "鬼武道",
    "蒙德高塔·第一卷",
    "与神性同行·序言",
    "列王与宗室史·序",
    "山与海之书",
    "玑衡经",
    "石书辑录·卷一",
]

def random_book_name():
    return random.choice(book_names)