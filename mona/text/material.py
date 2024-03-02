import random

'''
0. 怪物掉落
1. 野外直接采集
2. 野生生物
4. 宝箱掉落
    经验书
    武器矿
    角色天赋材料
    1-3星武器(放隔壁weapons里面了)
    1-4星圣遗物(放隔壁圣遗物里了)    
'''     

material_names = [
    # 怪物掉落
    '破损的面具', 
	'污秽的面具', 
	'不祥的面具', 

	'寻宝鸦印', 
	'藏银鸦印', 
	'攫金鸦印', 

	'骗骗花蜜', 
    '微光花蜜', 
	'原素花蜜', 

	'新兵的徽记', 
	'士官的徽记', 
	'尉官的徽记', 

	'浮游干核', 
	'浮游幽核', 
	'浮游晶化核', 

	'导能绘卷', 
	'禁咒绘卷', 
	'封魔绘卷', 

	'锐利的箭簇', 
    '牢固的箭簇', 
	'历战的箭簇', 

	'蕈兽孢子', 
	'荧光孢粉', 
	'孢囊晶尘', 

	'史莱姆凝液', 
	'史莱姆清', 
	'史莱姆原浆', 

	'褪色红绸', 
	'镶边红绸', 
	'织金红绸', 

	'破旧的刀镡', 
	'影打刀镡', 
	'名刀镡', 

	'雾虚花粉', 
	'雾虚草囊', 
	'雾虚灯芯', 

	'猎兵祭刀', 
	'特工祭刀', 
	'督察长祭刀', 

	'失活菌核', 
	'休眠菌核', 
	'茁壮菌核', 

    # 进行缩短
	'来自何处的待放之花', 
    '何人所珍藏之花', 
	'漫游者的盛放之花', 

	'脆弱的骨片', 
	'结实的骨片', 
	'石化的骨片', 

    '混沌机关', 
	'混沌枢纽', 
	'混沌真眼', 

	'沉重号角', 
    '黑铜号角', 
	'黑晶号角', 

	'晦暗刻像', 
	'幽邃刻像', 
	'夤夜刻像', 

	'破缺棱晶', 
	'混浊棱晶', 
	'辉光棱晶', 
    
	'隐兽指爪', 
	'隐兽利爪', 
	'隐兽鬼爪', 

	'地脉的旧枝', 
	'地脉的枯叶', 
	'地脉的新芽', 

	'混沌容器',
	'混沌模块',
	'混沌锚栓',

	'残毁的横脊', 
	'密固的横脊', 
	'锲纹的横脊', 

	'混沌装置', 
	'混沌回路', 
	'混沌炉心', 

	'黯淡棱镜', 
	'偏光棱镜', 
	'水晶棱镜', 
    
	# 4.0 枫丹新怪物
	'异海凝珠',
    '异海之块',
    '异色结晶石',
    
	'啮合齿轮',
    '机关正齿轮',
    '奇械机芯齿轮',
    '奇械机芯齿',
    
	'浊水的一滴',
    '浊水的一掬',
    '初生的浊水幻灵',

    
	'隙间之核',
    '外世突触',
    '异界生命核',
    
	# 4.1 新怪物
	"老旧的役人怀表",
    "役人的制式怀表",
    "役人的时时刻刻",
    
	# 4.2 新怪物
	"无光丝线",
    "无光涡眼",
    "无光质块",

    # ----------------
    # 怪物掉落结束

    # 野外直接采集
    ## 地区特产
    # 蒙德
    "小灯草",
    "慕风蘑菇",
    "蒲公英籽",
    "钩钩果",
    "落落莓",
    "风车菊",
    "嘟嘟莲",
    "塞西莉亚花",

    # 璃月
    "星螺",
    "绝云椒椒",
    "琉璃袋",
    "夜泊石",
    "石珀",
    "霓裳花",
    "琉璃百合",
    "清心",
    "清水玉",

    # 稻妻
    "晶化骨髓",
    "血斛",
    "海灵芝",
    "天云草实",
    "绯樱绣球",
    "珊瑚真珠",
    "鸣草",
    "幽灯蕈",
    "鬼兜虫",

    # 须弥
    "劫波莲",
    "悼灵花",
    "帕蒂沙兰",
    # "圣金虫"
    "赤念果",
    "月莲",
    "沙脂蛹",
    "树王圣体菇",
    "万相石",
    
	# 枫丹
	"柔灯铃",
    "虹彩蔷薇",
    "苍晶螺",
    "海露花",
    '茉洁草',
    "幽光星星",
    "子探测单元",
    "湖光铃兰",
    "初露之源",
    
    # 野生生物
	"红莲蛾", # 也算吧，不然不会自己拿了by default
    "蝴蝶",
    "黑背鲈鱼",
    "蓝鳍鲈鱼",
    "黄金鲈鱼",
    "风晶蝶",
    "雷晶蝶",
    "草晶蝶",
    "冰晶蝶",
    "岩晶蝶",
    "水晶蝶",
    "珊瑚蝶",
    "落日鳅鳅",
    "金鳅鳅",
    "晴天鳅鳅",
    "青蛙",
    "泥蛙",
    "蓝蛙",
    "丛林树蛙",
    "黄金蟹",
    "太阳蟹",
    "海蓝蟹",
    "将军蟹",
    "薄红蟹",
    "蓝角蜥",
    "红角蜥",
    "绿角蜥",
    "嗜髓蜥",
    "赤尾蜥",
    "藤纹陆鳗鳗",
    "深海鳗鳗",
    "赤鳍陆鳗鳗",
    "流沙鳗鳗",
    "吉光虫",
    "圣金虫",
	'萤火虫',

    # 宝箱掉落
	'流浪者的经验', 
	'冒险家的经验', 
    '大英雄的经验', 
	'精锻用杂矿', 
    '精锻用良矿', 
	'精锻用魔矿', 


	'「自由」的哲学', 
	'「抗争」的哲学', 
	'「诗文」的哲学', 
	'「繁荣」的哲学', 
	'「勤劳」的哲学', 
	'「黄金」的哲学', 
	'「风雅」的哲学', 
	'「浮世」的哲学', 
	'「天光」的哲学', 
	'「笃行」的哲学', 
	'「诤言」的哲学', 
	'「巧思」的哲学', 
	'「自由」的指引', 
	'「抗争」的指引', 
	'「诗文」的指引', 
	'「繁荣」的指引', 
	'「勤劳」的指引', 
	'「黄金」的指引', 
	'「风雅」的指引', 
	'「浮世」的指引', 
	'「天光」的指引', 
	'「笃行」的指引', 
	'「诤言」的指引', 
	'「巧思」的指引', 
	'「自由」的教导', 
	'「抗争」的教导', 
	'「诗文」的教导', 
	'「繁荣」的教导', 
	'「勤劳」的教导', 
	'「黄金」的教导', 
	'「风雅」的教导', 
	'「浮世」的教导', 
	'「天光」的教导', 
	'「笃行」的教导', 
	'「诤言」的教导', 
	'「巧思」的教导',
    # 4.0 新天赋书 
	'「公平」的哲学', 
	'「正义」的哲学', 
	'「秩序」的哲学', 
	'「公平」的指引', 
	'「正义」的指引', 
	'「秩序」的指引', 
    '「公平」的教导',
    '「正义」的教导',
    '「秩序」的教导',
    
    
	# 化种匣
	'「金鱼草」的种子',
    '「鸣草」的种子',
    '「清心」的种子',
	'「小灯草」的种子',
    '「香辛果」的种子',
    '「白萝卜」的种子',
    '「蘑菇」的孢子',
    '「绝云椒椒」的种子',
    '「马尾」的种子',
    '「墩墩桃」的种子',
    '「甜甜花」的种子',
    '「海灵芝」的样本',
    '「胡萝卜」的种子',
    '「莲蓬」的种子',
    '「须弥蔷薇」的种子',
    '「风车菊」的种子',
    # '「梦里花·星槿」的花种',
    '「薄荷」的种子',
    '「海草」的种子',
    '「嘟嘟莲」的种子',
    '「琉璃百合」的种子',
    '「琉璃百合」的',
    # '「梦里花·棠铃」的花种',
	'「霓裳花」的种子',
    '「琉璃袋」的种子',
	'「塞西莉亚花」的种子',
    '「落落莓」的种子',
    # '「梦里花·绣荚」的花种',
    '「久雨莲」的种子',
    '「茉洁草」的种子',
    '「柔灯铃」的种子',
    '「虹彩蔷薇」的种子',

    # 杂项
	'鱼肉',
    '铁块', 
	'兽肉', 
	'甜甜花', 
	'胡萝卜', 
	'蘑菇', 
	'松茸', 
	'禽肉', 
	'松果', 
	'土豆', 
	'金鱼草', 
	'莲蓬', 
	'薄荷', 
	'卷心菜', 
	'番茄', 
	'洋葱', 
	'鸟蛋', 
	'树莓', 
	'小麦', 
	'白萝卜', 
	'嘟嘟莲', 
	'白铁块', 
	'水晶块',
	'萃凝晶',
	'马尾', 
	'冰雾花', 
	'烈焰花', 
	'电气水晶', 
	'苹果', 
	'日落果', 
	'竹笋', 
	'星银矿石', 
	'「冷鲜肉」', 
	'紫晶块', 
	'海草', 
	'堇瓜', 
	'星蕈', 
	'墩墩桃', 
	'须弥蔷薇', 
	'香辛果', 
	'枯焦的星蕈', 
	'活化的星蕈', 
	'毗波耶', 
	'枣椰',
    '久雨莲',
    '泡泡桔',
    '汐藻',
	'神秘的肉', 
	'魔晶块', 
	'奇异的「牙齿」', 
	'汲取了生命力的龙牙',
    
	# 每日委托
	"新鲜的鸟蛋", # 全能美食队 烹饪对决
    "新鲜的绯樱绣球",
    "新鲜的海草",
    "新鲜的蘑菇",
    "新鲜的金鱼草",
    "新鲜的清心", # B小雀儿
    "纤维型海露花",
    "新鲜的绝云椒椒",
    "新鲜的树莓",
    "新鲜的香辛果",
    
	# 世界任务
	"纯洁松果",
    "碎星铁矿",

    # 1-3星武器

]
def random_material_name():
    return random.choice(material_names)
