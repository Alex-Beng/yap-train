import random
import os
import json
import pickle

import cv2
from PIL import Image
import lzma
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
_init = False
bg_imgs = []

def load_bg_imgs(path = "./dumps_full_mona2/"):
    # check path exists
    if not os.path.exists(path):
        print(f'path {path} not exists')
        print(f'trying other path')
        return load_bg_imgs("../yap/dumps_full_mona2/")
    # 获取文件夹下所有图片
    files = os.listdir(path)
    # 读取图片
    imgs = []
    for file in files:
        _timage = cv2.imread(path + file)
        # 如果col < 384, 则resize
        if _timage.shape[1] < 384:
            row, col = _timage.shape[:2]
            new_row, new_col = row/col*384, 384
            _timage = cv2.resize(_timage, (int(new_col+1), int(new_row+1)))  
        imgs.append(_timage)
        # 反转
        imgs.append(cv2.flip(_timage, 1))
        # 加个高斯噪声
        imgs.append(cv2.GaussianBlur(_timage, (21, 21), 0))
        # 椒盐噪声
        imgs.append(cv2.blur(_timage, (21, 21)))
    print(f'load {len(imgs)} bg imgs')
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
    "猫咪事务所·卷一",
    "猫咪事务所·卷二",
    "猫咪事务所·卷三",
    "猫咪事务所·卷四",
    "猫咪事务所·卷五",
    "猫咪事务所·卷六",
    "猫咪事务所·卷七",
    "猫咪事务所·卷八",
    "小魔女与不熄灭的火·卷一",
    "小魔女与不熄灭的火·卷二",
    "小魔女与不熄灭的火·卷三",
    "小魔女与不熄灭的火·卷四",
    "小魔女与不熄灭的火·卷五",
    "小魔女与不熄灭的火·卷六",
    "小魔女与不熄灭的火·卷七",
    "造访神秘房间",
    "幻想真境剧诗",
    "抽取本期签文", "抽取本期签文", "抽取本期签文", "抽取本期签文", "抽取本期签文", "抽取本期签文",
    "更改旋律",
    "准备演出",
    "继续演出",
        "","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",
        "《遗珑埠不容错过的三种小吃》",
        "《遗珑埠不容错过的三种小吃》",
        "《遗珑埠不容错过的三种小吃》",
        "《遗珑埠不容错过的三种小吃》",
        "《遗珑埠不容错过的三种小吃》",
        # "临瀑之城",
        # "加入奶泡",
        # "《遗珑埠不容错过的三种小吃》",
        # "决定上午的安排",
        # "清水玉",
        
        # "残毁的剑柄",
        # "裂断的剑柄",
        # "未熄的剑柄",
	# 5.0 新怪物
    "意志破碎的残片",
    "意志明晰的寄偶",
	"意志巡游的符像",

	"聚燃的石块",
    "聚燃的命种",
	"聚燃的游像眼",

	"秘源轴",
    "秘源机鞘",
	"秘源真芯",
    
	"稚嫩的尖齿",
    "老练的坚齿","老练的坚齿","老练的坚齿","老练的坚齿","老练的坚齿",
	"横行霸者的利齿",

        "青蜜莓",
    "青蜜莓",
    "青蜜莓",


	"卫从的木哨","卫从的木哨","卫从的木哨","卫从的木哨","卫从的木哨","卫从的木哨","卫从的木哨",
    "卫从的木哨","卫从的木哨","卫从的木哨","卫从的木哨",
    "战士的铁哨","战士的铁哨","战士的铁哨","战士的铁哨",
	"龙冠武士的金哨","龙冠武士的金哨","龙冠武士的金哨","龙冠武士的金哨",
	# 纳塔
	"颗粒果",
    "烛伞蘑菇",
    "烬芯花",
    "澄晶实",
    "苦种",
    "肉龙掌",
    "灼灼彩菊",
    "浪沫羽鳃",
        # 5.0 新天赋书
	'「角逐」的哲学', 
	'「焚燔」的哲学', 
	'「纷争」的哲学', 
	'「角逐」的指引', 
	'「焚燔」的指引', 
	'「纷争」的指引', 
    '「角逐」的教导',
    '「焚燔」的教导',
    '「纷争」的教导',
    "牛奶",
    "糖",

    "固晶甲虫","固晶甲虫","固晶甲虫",
    "飞飞","飞飞","飞飞",

# "我来猜猜",
# "是元能构装体？",
# "是水深渊法师？",
# "是穿蓝衣服的小兔子？",
# "是超大松鼠？",
# "是短鬓虎？",
# "是穿绿衣服的狐狸？",
# "我来试试",
# "关于巨龙…",
# "你还好吗？",
# "魔线飞行",
# "老旧的日记",
# "发黄的信件",
# '奥比奇',
# '波波拉诺',
# '温柔的守卫',
# '藏宝爱好者',
# '卡佩',
# '艾德加',
# '席尔万',
# '卢西恩',
# '进入席尔万房间',
# '进入卢西恩房间',
# '进入艾德加房间',
# # '议论纷纷的群众',
# # '议论纷纷的群众',
# # '议论纷纷的群众',
# # '议论纷纷的群众',
# # '议论纷纷的群众',
# '议论纷纷的群众',
# '进入席尔万房间',
# '艾德加',
# '卡佩',
# '奥比奇',
# '波波拉诺',
# '探长',
# '埃斯诺尔',
# '帕纽尔',
# '罗泽尔',
# '优雅的雕像',
# '用力推！！！',
# '用力拽！！！',
# '用力抬！！！',
# '温和的雕像',
# '平静的雕像',
# '福罗贝尔',
# '奥雷卢',
# '埃斯唐普',
# '西梅',
# '萨博兰',
# '苏法什',
# '陈旧的手稿',
# '疲惫的手稿',
# '兴奋的纸蛙',
# '沉稳的纸蛙',
# '好心的纸蛙',
# '忧郁的纸蛙',
# '123木头人',
# '123木头人2',
# '门后的声音',
# '注脚',
# '巨人卫兵',
# '路人',
# '星轨王城守卫',
# '店长',
# '粮油店长',
# '巨人卫兵',
# '巨人卫兵',
# '巨人卫兵',
# '巨人卫兵',
# '星轨王城守卫',
# '肖维涅',
# '我来试试',
# '我来试试',
# '我来试试',
# '我来试试',
# '我来试试',
# '我来试试',
# '巨人卫兵',
# '巨人卫兵',
# '埃斯诺尔',
# '失望的纸蛙',
# '不满的纸蛙',
# '发生什么了？',
# '字迹歪歪扭扭的信',
# '杏仁',
# '溪泉',
# '柑橘',
# '翻看书页',
# '布勒凡',
# '布勒施',
# '欧立威',
# '毕塞尔特',
# '搭乘「空中列车」',
# '杜林',
# '小杜林',
# '拜希麦·五世',
# '爆竹',
# '板栗',
# '白果',
# '瑟杜尔',
# '威尔',
# '积木卫兵',
# '蒙塔纳',
# '魔女M',
# '魔女B',
# '冲在前面的士兵',
# '冲在前面的士兵',
# '门内的人',
# '门内的人',
# '门内的人',
# '飞鼠',
# '士兵',
# '士兵',
# '小飞鼠',
# '飞鼠妈妈',
# '老旧的日记',
# '发黄的信件',
# '进入「邪龙」的巢穴',
# '德维特',
# '许迪',
# '克朗谢',
# '埃洛夫',
# '杏仁',
# '溪泉',
# '柑橘',
# '等待海上列车…',
# '前往提瓦特',
# '前往星轨王城',
# '前往破碎之海',
# '前往提瓦特',
# '「一篇未完成的手稿」',
# '杏仁',
#         "祀珑典仪",
#             "旋转管道",
#     "让魔水流动吧！",
#     "就是现在",
#     "敲一敲…",
#     "我带你飞一次？",
#     "我来帮忙！",
#         "戳一戳",
#         "出发吧！",
#         "有点奇怪…",
#     "拉你一把",
#     "发生什么了？",
#     "交给我吧",
#     "已经安全了！",
#     "我需要那颗星星",
#     "用力拉",
#         "一起用力拉",
#         "你已经做得很好了",
#         "搭乘「空中列车」",
#         "这是什么比赛？",
#             "与纠结的小人结队",
#     "与贪吃的小人结队",
#     "与莽撞的小人结队",
#     "开始挑战",
#     "放置蘑菇",
#     "放置苹果",
#     "放置风车菊",
#     "放置鸟蛋",
#     "放置树莓",
#     "放置禽肉",
#     "放置蘑菇",
#     "放置甜甜花",
#     "放置绝云椒椒",
#     "走左边",
#     "走右边",
#     "走前面",
#     "算我一个！",
#     "找到你啦！",
#     "等待海上列车…",
#     "拾取",
#     "放置",

#         "开始演奏",
#     "募刻巧像",
#     "绘想游迹",
#     "召唤激流",
#     "放入",
#     "推一下",
#     "摆放巧像",
#     "准备留影",
#     	# 4.8
# 	"「永远高挂的落日」",
#     "「岸上游泳的鱼儿」",
#     "「夜晚显现的明月」",
    
# '小狼',
# '拉瓦特',
# '奥利亚克',
# '伊丝翠',
# '阿奈',
# '顾客们',
# '博格诺',
# '安蒂布',
# '奥德丽',
# '贝尔特朗',
# '病重的老太太',
# '「巫婆」',
# '杜邦',
# '莫嘉娜',
# '希娅',
# '杜庞',
# '埃里克',
# '波顿',
# '维耶',
# '奥诺雷',
# '奥迪伦',
# '朗贝萨',
# '加斯东',
# '昂利',
# '拉扎尔',
# '查看桌台',
# '查看相框',
# '查看置物柜',
# '吕克',
# '阿托莎',
# '弗洛莱恩',
# '忧愁的女子',
# '蛮横的商贩',
# '焦虑的男子',
# '卫兵队长',
# '受伤的卫兵',
# # '受伤的卫兵',
# # '受伤的卫兵',
# # '受伤的卫兵',
# # '受伤的卫兵',
# '驼背的老妇人',
# '严肃的卫兵',
# '王国首相',
# '库尔布瓦',
# '（观察）',
# # '（观察）',
# # '（观察）',
# # '（观察）',
# # '（观察）',
# # '（观察）',
# # '（观察）',
# '弗洛莱恩',
# '弗洛莱恩',
# '弗洛莱恩',
# '「拉扎尔」',
# '病重的老太太',
# '门上的公告',
# '生产日志',
# '实验日志',
# '成女路人',
# '老人路人',
# '小孩路人',
# '成男路人',
# '「昂利」',
# '兽境幼犬',
# '兽境幼犬',
# '成女路人',
# '小孩路人',
# '成男路人',
# '弗洛莱恩',
# '布蕾斯特',
# '苏定',
# '玄理',
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
'《提瓦特世界史·第一部分》',
'《枫丹香水广告集》',       
'《白猫神探》',
'《薰衣草之花》',
'「伊利萨」',
'「半熟蛋」',
'「合成台」',
'「咕噜噜」',
'「大椰子」',
'「小波」',
'「此地的伟大精神」',       
'「毛发如夜的野兽」',       
'「灵道刻印门」',
'「炫彩呆子」',
'「狂图捕手」',
'「绿松石」',
'「翡翠」',
'「西弗安雅」',
'「金盏花」',
'「队长」',
'「黑曜石碑的影像」',       
'丁戈内克',
'万奎洛',
'万杰鲁',
'不耐烦的贼匪',
'不耐烦的部族成员',
'专注的「花羽会」战士',
'为伙伴们烹饪料理',
'乌丘',
'乌兰塔',
'乌图伦库',
'乌尔',
'乌米',
'乌苏坎',
'乌茨勒',
'互动点',
'亚格贝乌',
'亚瓦尔',
'亚蕾琪',
'代杜古',
'伊克娜',
'伊兹娜',
'伊卡力',
'伊卡尤',
'伊图提娅',
'伊妮',
'伊扎帕',
'伊法',
'伊波莱',
'伊泽',
'伊瓦伊',
'伊约芭',
'伊茨昆',
'伊莫莎',
'伊莱帕',
'伊蒂莉',
'伊蕾尔',
'伊蜜',
'伊西多尔',
'伊西姆',
'伊赫卡',
'众人写成的记录',
'低语',
'佩蕾·努伊',
'依扎娜',
'依稀可以辨认的布告',
'克伊奈',
'克拉莉希娅',
'克瓦斯尼科夫',
'克维库',
'兰卡亚',
'兰维达',
'兰萨提',
'兰齐塔',
'兴奋的「回声之子」战士',
'冰冷的身躯',
'冷静的「回声之子」战士',
'凯塔',
'凯安',
'出魂试炼之所的留言',
'切妮',
'列扎诺夫',
'利别科',
'利洛',
'加勒弗大汪',
'加塔力',
'加杰雷',
'加纳尔',
'努乌',
'努斯妲',
'勇敢的匿叶龙',
'勇猛的嵴锋龙',
'勒拉',
'匿叶龙',
'南娜',
'卡乌帕',
'卡乌胡',
'卡乌鲁',
'卡伦雅',
'卡克斯',
'卡克斯的行装',
'卡兹加',
'卡力弗',
'卡卡妈妈',
'卡嘉',
'卡图马',
'卡安',
'卡帕可',
'卡拉穆鲁',
'卡纳鲁亚',
'卡胡那',
'卡胡鲁',
'卡马克',
'卡马加',
'卡齐娜',
'受困的匿叶龙',
'受困的嵴锋龙',
'古塞尼科夫',
'古寻梦祭司的记录·其之一',
'古寻梦祭司的记录·其之三',
'古寻梦祭司的记录·其之二',
'古寻梦祭司的记录·其之四',
'古瑟雷德',
'古老的布告',
'古老的日记',
'古老的碑刻',
'古雷马',
'叶莱菲',
'吉恩布',
'向下看',
'呆呆的小嵴锋龙',
'呆呆的绵驮兽',
'呼噜克',
'哈卡乌',
'哈蒂丝',
'哈隆加',
'唤灵师的手记·其一',
'唤灵师的手记·其二',
'商队成员',
'嗜岩·兽境幼兽',
'嘉乌兰妮',
'因卡纳克',
'因娜',
'团雀01',
'团雀02',
'团雀03',
'困倦的绒翼龙',
'图尼娜',
'图帕克',
'图蒂坦',
'图蕾娅',
'图达玛',
'圣王的手记·其一',
'圣王的手记·其七',
'圣王的手记·其三',
'圣王的手记·其九',
'圣王的手记·其二',
'圣王的手记·其五',
'圣王的手记·其八',
'圣王的手记·其六',
'圣王的手记·其四',
'坎内克',
'埃文苏',
'基尼奇',
'基斯庇',
'基班古',
'基维拉',
'基耶霍',
'塔力帕',
'塔卢加',
'塔尔科',
'塔恩加',
'塔洛娜',
'塔穆因',
'塔莫蕾',
'塔诺',
'塞奇托',
'塞巴尔',
'夏安卡',
'夏拉帕',
'夏温',
'夏芙米娅',
'夏萝·努伊',
'多巴',
'夜猫子',
'夜神',
'夜魂之灵',
'大家伙？',
'大胆的调查员',
'夸恩因',
'夸恩因的日记',
'夸梅',
'奇卡',
'奇怪的匿叶龙',
'奇普',
'奇梅',
'奇玛',
'奇科娅',
'奇马利',
'奈赫西',
'奎兹',
'奎泰克',
'奎皮亚',
'奥伯龙',
'奥坎比',
'奥奇坎',
'奥奇坎的残躯',
'奥娅',
'奥尔科',
'奥希娜',
'奥杜力',
'奥科兰',
'奥罗卢柯',
'奥西',
'奥贡',
'好奇的调查员',
'妮乌',
'妮佩卡',
'妮赫佳',
'威严的低语',
'娅特兰',
'娜恩希',
'娜玛卡',
'子澄',
'字迹模糊的笔记',
'孟尼力克',
'安切塔',
'安萨鲁',
'安蒂妮',
'家长般的大嵴锋龙',
'容韦勒',
'对话群演',
'小孩的笔记',
'小家伙？',
'小小蜜虫',
'尤季娜',
'尤庇娅',
'尤考姆',
'尼南',
'尼尼',
'尼赞贝',
'山王',
'嵴锋龙',
'嵴锋龙DJ',
'嵴锋龙斗舞女1',
'嵴锋龙斗舞女2',
'嵴锋龙斗舞男1',
'嵴锋龙斗舞男2',
'巡山客的手记',
'工匠的笔记·其一',
'工匠的笔记·其二',
'巴亚吉达',
'巴克',
'巴加约',
'巴哈蒂',
'巴拉诺夫',
'巴茨',
'布基纳',
'布尔莎',
'布纳马',
'布鲁',
'希乌休',
'希亚赫',
'希伊亚',
'希巴拉克',
'希库埃鲁',
'希皮利',
'希诺宁',
'帕兰',
'帕加尔',
'帕卡·努伊',
'帕尼娅',
'帕恰',
'帕朗',
'帕莉西娅',
'帕贾丹',
'年轻的行商',
'年长的行商',
'幻写灵',
'幼光',
'幼小的匿叶龙',
'幼嵴锋龙',
'幼年的暝视龙',
'庇兰',
'库伊米',
'库基亚',
'库娅',
'库希马纳',
'库延',
'库拉霍',
'库斯可',
'库特比',
'库珀·努伊',
'库约尼',
'库纳克',
'库西',
'库阿勒',
'库马约',
'庞塞',
'废弃-奇玛',
'废弃-希乌休',
'废弃-曼扎那',
'废弃-森塔加',
'废弃-阿乌伊娅',
'废弃-马察特利',
'延古',
'延塔莎',
'延索',
'待物3带背包',
'待物3带背包 深肤色',
'德乌特',
'德乌特（灵）',
'忧愁的绒翼龙',
'怪物2带背包',
'怪物2带背包 深肤色',
'怪物带背包',
'怪物带背包 深肤色',
'怯生生的小匿叶龙',
'恩加韦',
'恩古迪',
'恩孔加',
'恩托提亚',
'恩琳',
'恰斯卡',
'悠然的「花羽会」战士',
'愤怒的贩龙者',
'懒懒的鳍游龙',
'戈利科夫',
'战场情报',
'扎安娅',
'托奥',
'托尔特克',
'托托',
'托扬娜',
'托泰克',
'托赞',
'托赫兰妮',
'拉坎阿',
'拉密萨纳',
'拉斯塔恩',
'拉曼',
'拉比娜',
'拉马奈',
'挑战起点',
'振尘',
'提佐科',
'提凯伊',
'提帕诺',
'提莉乌娅',
'搭建帐篷',
'摩亚瓦',
'放置「古名」',
'斯塔杜欣',
'斯普赫',
'无名工匠的笔记',
'无名旅人的笔记·其一',
'无名旅人的笔记·其三',
'无名旅人的笔记·其二',
'无名旅人的笔记·其五',
'无名旅人的笔记·其四',
'无名祭司的笔记',
'昆纳乌',
'晕倒的匿叶龙',
'晕倒的鳍游龙',
'普玛',
'普玛莉',
'暝视龙',
'暴躁的部族战士',
'更换衣服',
'曼尤雅',
'曼扎那',
'曼科',
'曼萨',
'最佳飞行成绩记录',
'朗戈',
'木箱',
'未完成的乐稿',
'朱南',
'机灵的小嵴锋龙',
'来自过去的声音',
'松伽塔',
'某人留下的铭记·其一',
'某人留下的铭记·其三',
'某人留下的铭记·其二',
'某人留下的铭记·其五',
'某人留下的铭记·其四',
'某人遗落的风镜',
'柯奇妲',
'柯娅',
'柯林伍德',
'柯梯米',
'树上的小嵴锋龙',
'栖木普',
'格罗贝尔',
'格茜妮',
'格里斯潘',
'桑哈杰',
'桑科约克',
'梅兹莉',
'梅内娅',
'梅莎',
'梅西娜',
'森塔加',
'森珀娅',
'樋口铁矢',
'欣赏景色',
'欧洛伦',
'欧莎薇',
'残破的石碑',
'比于什',
'比奇津',
'比赛公告板',
'比赛选手',
'毛球',
'水豚',
'汉佩克',
'汉扎',
'沃蕾',
'波卡',
'波吉吉',
'波奇奇',
'波尼尼',
'波蒂莉',
'波蒂薇',
'泽伊拉',
'洛娅',
'活泼的绵驮兽',
'活跃的小嵴锋龙',
'海兹尔',
'海岳',
'涅兰',
'涅提',
'游学者的笔记',
'潘帕',
# '灰色灵体女儿（表演用灰色人1）',
# '灰色灵体女儿（表演用灰色人2）',
# '灰色灵体女儿（表演用灰色人3）',
# '灰色灵体女儿（表演用灰色人4）',
# '灰色灵体女儿（表演用灰色人5）',
'灵体真女儿',
'爱可',
'爱梅拉',
'爱维莉',
'爱菲拉',
'爱雅',
'特佐佐',
'特佩潘',
'特帕切',
'特帕尔',
'特拉佐莉',
'特拉佐莉（表演用克隆人）',
'特拉帕尔',
'特拉忒',
'特拉波',
'特拉科',
'特沃德罗斯',
'特波兹',
'特立尼达',
'特纳华',
'特蒂赫莉',
'特诺切',
'特阿墨',
'特阿尤',
'特马乔',
'狄雷',
'献上供物',
'玛亚薇',
'玛伊蕾',
'玛兹嘉',
'玛拉妮',
'玛薇卡',
'玻娜',
'琪丝佩',
'琪希可',
'瓦伊妲',
'瓦伊纳',
'瓦兰卡',
'瓦凯·努伊',
'瓦卡尔',
'瓦图妲',
'瓦尔米',
'瓦托娅',
'瓦斯卡尔',
'瓦曼',
'瓦蕾',
'瓦达娜',
'界碑上的刻文',
'疑惑的调查员',
'疲惫的暝视龙',
'白浜',
'皮尔钦',
'皮希加',
'皮萨克',
'看向仪式台',
'研究中心紧急机械操作手册',
'磨损的记录',
'秋纽奥',
'科优尔',
'科奥劳',
'科奴拉芙',
'科帕',
'科斯特利茨基',
'科潘',
'科科庞',
'科米娜',
'科约帕',
'科莫奎',
'秘源哨机',
'穆伊萨',
'穆基',
'穆尔科',
'穆托塔',
'穆纳伊',
'穆蒲',
'竖立在旁的木牌',
'等待…',
'米卢·努伊',
'米奎克',
'米特尔',
'米齐卡',
'索尔涅夫',
'索莉马娜',
'约洛特莉',
'练习钻地的小嵴锋龙',
'细心的匿叶龙',
'绒翼龙',
'绒翼龙幼龙',
'维兰德',
'维图索夫',
'维护记录·其一',
'维护记录·其三',
'维护记录·其二',
'维查玛',
'维莎',
'羊驼',
'群演1',
'老鼠',
'耶里门',
'胆小的绵驮兽',
'胡妮',
'胡里约',
'自称「山下」之物',
'艾库瓦',
'苏娅',
'苏玛克',
'若娜瓦',
'茜特菈莉',
'荆和',
'莉乌娅',
'莉洛娅',
'莉琉',
'莉瓦',
'莉莉尤',
'莎乌莎',
'莎娜',
'莎维娅',
'莎迪蕾',
'莫乌',
'菲欣妮',
'菲耶蒂娜',
'萨加图拉',
'萨卡乌',
'萨科巴',
'萨约尔',
'萨费伊',
'葵可',
'葵莉娅',
'蒂亚戈',
'蒂图',
'蒂尔妲',
'蒂索克',
'蓬普',
'薇尔恰',
'虚弱的声音',
'虚弱的大灵显化身',
'虚弱的绒翼龙',
'虚弱的膨膨兽',
'被困的匿叶龙',
'被困的嵴锋龙',
'被困的幼暝视龙',
'被救出的匿叶龙',
'被救的鳍游龙',
'装饰花车',
'西坎',
'西布',
'西帕克',
'西蒂娜',
'西薇',
'观察护身符',
'观察武器',
'观察罐子',
'观察腰带',
'观察钓具',
'警示者的记录',
'记录用秘源哨机',
'谁人的手记',
'调查用投喂点',
'谜画涂鸦中的意志',
'谨慎的调查员',
'贝坎',
'贝雷哈努',
'贤龙',
'贪吃的绵驮兽',
'赫伯德',
'赫瓦特莉',
'辛佳',
'辛奇',
'达伊妲',
'达尔戈',
'达布亚',
'达祖玛',
'进行比赛',
'迷惑的调查员',
'迷灰中的石碑·其一',
'迷灰中的石碑·其三',
'迷灰中的石碑·其二',
'通皮斯',
'醒目的标语',
'钦特利',
'闪亮亮的贝壳',
'阿乌伊娅',
'阿乌琪',
'阿乔',
'阿伽娅',
'阿凯蒂',
'阿卡坦加',
'阿卡马',
'阿图科',
'阿坎蒂',
'阿基约',
'阿奇亚库',
'阿妮拉',
'阿姆鲁',
'阿孔',
'阿尔妲',
'阿尔帕',
'阿尤',
'阿布帕隆',
'阿希莉',
'阿斯基娅',
'阿斯麦隆',
'阿格莫',
'阿梅妲',
'阿特琪',
'阿科米蒂',
'阿米娜',
'阿纳',
'阿蕾梅',
'阿达玛',
'阿邦迪',
'阿隆',
'阿霍伊图',
'阿马佐利',
'阿鲁吉',
'陀罗诘利',
'陀达罗',
'难过的嵴锋龙',
'雅库普',
'雅蕾希',
'雅诺斯基',
'雇主',
'雷格巴',
'霍图鲁',
'霍娅',
'霍莱',
'韦尼埃米诺夫',
'顾客',
'马伊塔',
'马兹拉',
'马凯莫',
'马努阿',
'马塔伽',
'马塔伽！记得！！',
'马塔拉',
'马察特利',
'马尼菲科',
'马扎坦',
'马提库伊',
'马洛考',
'马特兰',
'马穆勒',
'马蒂亚斯',
'马西娜·努伊',
'马陶霍',
'驻扎日记',
'验收与维修记录',
'鳍游龙',
'鳍游龙夏威夷吉他',
'鳍游龙尤克里里',
'鳍游龙木吉他',
'鳍游龙箱鼓',
'麦斯欧德',
'黛希特莉',
'龙铭刻',
'（翻找箱子…）',
'「煤球」',
'「花羽叶月神机·辉翼型」',
'万普纳',
'乌卡曼',
'乔甘',
'伊拉德',
'伊柯蒂',
'伊露妮',
'众人齐声',
'佩尤特莉',
'佩索妲',
'前往「古斯托特」所在之处',
'劳塔洛',
'卡蒂佳',
'卡赫伊娅',
'卢塔',
'厄切敏',
'咔库库',
'哈尔乔',
'哈恩薇',
'哈恩薇&安帕奥',
'嘉伊帕',
'基奈',
'塞妲',
'奇恩延',
'奎德雷',
'奥菈雅',
'奥蕾帕',
'安宿',
'安帕奥',
# '对手A',
# '对手B',
'岩史莱姆',
'巴杜卢',
'希涅琪娅',
'帷锦',
'库库娅',
'库穆库尔',
'恩德盖',
'恩铁戈',
'成年暝视龙',
'戟冠鸟',
'托哈',
'投入的观众',
'明澜',
'有些怪异的「人」',
'松科',
'柯夏尼娜',
'格洛德克',
'梅索',
'欣切尔',
'欧蒂菈',
'沃塔斯卡',
'沃杜奈',
'法坎佳',
'波鲁亚',
'洛塔',
'洛柯娅',
'派蒙&卡齐娜&玛拉妮',
'源火之圣座',
'热情的男子',
'爷爷',
'琪莎',
'瑶露',
'瓦内杜',
'瓦芒加',
'看看「祝柬」吧…',
'碧夸妮',
'祖马力',
'穆尔科&维查玛',
'穆托瓦',
'竞技场观众',
'等待海灯节来临...',
'维奇琳',
'维查莎',
'罗契耶夫',
'老爹',
'胡阿库',
'致敬',
'艾列尔',
'花羽会部族成员',
'莎柯薇',
'萨米诺',
'萨迪古亚',
'落水女子',
'落水女子B',
'落水男子',
'蒙吕松',
'蓝砚',
'蕾伊安',
'认真思考的女子',
'谭托',
'贝苏琪',
'赫卡',
'赫纳卡',
# '路人B',
# '路人C',
# '路人D',
'返回春香窑',
'选手',
'阿亚米',
'阿伊勒',
'阿依瓦丁',
'阿卡特',
'阿拉凯',
'阿芸',
'阿蒂克',
'雅塔',
'马索',
'鲁米帕',
'齐声',

    "折光的胚芽",
    "惑光的阔叶",
    "迷光的蜷叶之心"    "折光的胚芽",
    "惑光的阔叶",
    "迷光的蜷叶之心"    "折光的胚芽",
    "惑光的阔叶",
    "迷光的蜷叶之心"
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
    cv2.addWeighted(black2white, 0.3, res_img, 0.7, 0, res_img)
    # 补偿亮度
    refine_ratio = 1 / 0.8
    refine_ratio = min(refine_ratio, 255 / np.max(res_img))
    res_img = res_img.astype(np.float32) * refine_ratio
    res_img = res_img.astype(np.uint8)

    min_count_val = random.randint(white_thre//2+100, 255)
    # min_count_val = white_thre//2+100

    rand_img = np.full((32, 384), min_count_val, dtype=np.uint8)
    # img = cv2.addWeighted(rand_img, 0.2, img, 0.8, 0, img)
    # 将img中的白色像素点的值变为rand_img中的值，使用opencv的bitwise_and
    img = cv2.bitwise_and(rand_img, rand_img, mask=img)

    # 随机权重叠加字和背景
    wdg = random.uniform(0.3, 0.6)
    # wdg = 0.4
    res_img = cv2.addWeighted(res_img, wdg, img, 1-wdg, 0, res_img)

    # res_img 随机乘以一个系数
    max_pixel = res_img.max()
    max_ratio = 255 / max_pixel
    res_img = res_img * random.uniform(0.8, max_ratio)

    # 0.5 概率反色
    if random.random() < 0.5:
        res_img = 255 - res_img

    res_img = res_img.astype(np.uint8)

    res_img = Image.fromarray(res_img)
    return res_img, text



def js_dp(obj, path):
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

def js_ld(path):
    return json.load(open(path, 'r', encoding='utf-8'))

def on_init():
    global genshin_x_imgs, genshin_y, genshin_n
    global another_x_imgs, another_y, another_n
    global bg_imgs
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
    
    try:
        another_x_imgs = pickle.load(lzma.open('/media/alex/Data/another_x_imgs.pkl', 'rb'))
        another_y = pickle.load(lzma.open('/media/alex/Data/another_y.pkl', 'rb'))
    except:
        backup_path = "D:/"
        backup_x_path = os.path.join(backup_path, 'another_x_imgs.pkl')
        backup_y_path = os.path.join(backup_path, 'another_y.pkl')
        if os.path.exists(backup_x_path) and os.path.exists(backup_y_path):
            another_x_imgs = pickle.load(lzma.open(backup_x_path, 'rb'))
            another_y = pickle.load(lzma.open(backup_y_path, 'rb'))
        else:
            another_x_imgs = []
            another_y = []


    assert(len(genshin_x_imgs) == len(genshin_y))
    genshin_n = len(genshin_y)
    assert(len(another_x_imgs) == len(another_y))
    another_n = len(another_y) 
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

def generate_pickup_image(rand_func=random_text, ratios: list[int, int] = [0.33, 0.33]):
    global _init
    if not _init:
        on_init()
        _init = True
    assert len(ratios) == 2
    assert sum(ratios) < 1
    
    # 三部分数据，
    # 手工标注的 genshin， validate 中错误的 another，生成数据
    rand_num = random.random()
    if rand_num < ratios[0]:
        idx = random.randint(0, genshin_n - 1)
        text = genshin_y[idx]
        img = genshin_x_imgs[idx]

        img = Image.fromarray(img)
        return img, text
    elif rand_num < ratios[0] + ratios[1]:
        idx = random.randint(0, another_n - 1)
        text = another_y[idx]
        img = another_x_imgs[idx]

        img = Image.fromarray(img)
        return img, text
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
def generate_mix_image(pickup_rand_func=random_text, pickup_gaa_ratios: list[int, int]=[0.33,0.33], pickup_ratio=0.5):
    global _init
    if not _init:
        on_init()
        _init = True
    if random.random() < pickup_ratio:
        return generate_pickup_image(pickup_rand_func, pickup_gaa_ratios)
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
