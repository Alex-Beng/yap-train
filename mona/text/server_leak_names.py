import random


# 从泄漏服务端获取的名字
server_leak_names = [
"《名冒险家系列》书封",
"《层岩巨渊「灵石」调查书》",
"《层岩巨渊矿产图鉴》",
"《层岩巨渊矿产报告》",
"「一头菇」",
"「伊部」",
"「八宝」",
"「公义」",
"「刑部小判」",
"「千日喧哗大飞车」",
"「博士」",
"「吉法师」",
"「塞琉斯」",
"「大肉丸」",
"「安全距离爆破装置·信号指南」",
"「宝箱」",
"「寝子」",
"「左世」",
"「式大将」",
"「式小将」",
"「御签挂」",
"「心」之封印",
"「怪力三十三」",
"「怪鸟」",
"「愚人众」守卫",
"「愚人众」小队长",
"「我的宝物」",
"「昆布丸」",
"「本周膳食」",
"「本周轮值」",
"「法罗赫之子」的日志·其一",
"「法罗赫之子」的日志·其三",
"「法罗赫之子」的日志·其二",
"「海乱鬼」",
"「海乱鬼」头领",
"「玄冬林檎」",
"「电气鞘翅」",
"「紧急通知」",
"「纳尔吉斯」",
"「耕地机」",
"「胆小」的守卫",
# "「胆怯」的守卫4",
"「胆怯」的守卫",
"「致代理团长」的信",
"「致法尔伽团长」的信",
"「致艾莉丝女士」的信",
"「致阿贝多先生」的信",
"「船工」",
"「花角玉将」",
"「萝赞」",
"「蒲公英」",
"「记忆的铭文」",
"「证城」",
"「贪婪」的守卫",
# "「贪婪」的守卫3",
"「造笼师」",
"「金龟马戏番长」",
"「阿瑠」",
"「阿釜」",
"「风车菊」",
"一块古老的石碑",
"一封残破的信",
"一平",
"一张告示",
"一张纸条",
"一弦",
"一枚钥匙",
"一柱",
"一正",
"一然",
"一道",
"一阔",
"七夜由美",
"七尾",
"七郎",
"万文集舍告示",
"万货通",
"三岛道世",
"三河茜",
"三浦",
"三田",
"上杉",
"上野",
"上锁的箱子",
"上香",
"下沉遗迹钟声",
"丑角",
"丘丘人暴徒",
"东东",
"严严实实的藤蔓",
"严备",
"严重磨损的锄头",
"中西",
"中野",
"中野志乃",
"丹丹",
"丹吉尔",
"丹羽",
"丹迪",
"丽玛",
"乃亚卜",
"久保",
"久利由卖",
"久利须",
"久岐妙",
"久彦",
"久濑",
"久美",
"义坚",
"义高",
"乌代",
"乌姆",
"乌尔曼",
"乌尔法",
"乌尔班",
"乌帽子",
"乌森",
"乌纳因",
"乌维",
"乌达德",
"乐明",
"乔伊斯",
"乔姨",
"九条孝行",
"九条政仁",
"九条镰治",
"乾玮",
"乾玮的规划书",
"于哥",
"于嫣",
"于尔根",
"于连",
"于连的狗",
"云叔",
"云江",
"云淡",
"五百藏",
"井上",
"井口的闭锁",
"井手",
"亚卡巴",
"亚历山德拉",
"亚拉",
"亚琉",
"亮子",
"仁羽",
"今泉",
"今谷三郎",
"今谷佳祐",
"今谷香里",
"仔细观察山洞入口…",
"仔细观察画像…",
"代号「乌鲁兹」的记录",
"代号「安斯兹」的记录",
"代号「特里萨兹」的记录",
"代号「莱德赫」的记录",
"代号「菲弗」的记录",
"令人在意的石壁",
"仲林",
"伊万",
"伊代尔",
"伊凡",
"伊利亚斯",
"伊利亚斯妈妈",
"伊利亚斯妹妹",
"伊利亚斯爸爸",
"伊娜丝",
"伊娜姆",
"伊尔佳",
"伊庭",
"伊恩",
"伊扎德",
"伊拉兹",
"伊拉杰",
"伊桑",
"伊泽姆",
"伊萨米",
"伊达",
"伊达里",
"伊部",
"休息",
"众人",
"优丹",
"优午",
"传令兵",
"传次郎",
"伯恩哈德",
"伯桓子",
"伯特兰",
"伶一",
"伶俐鬼",
"伸夫",
"伽塔",
"伽禄",
"佐佐木",
"佐藤",
"佐赫蕾",
"佑旭",
"佣兵",
"佩佩",
"佩尔西科夫",
"供奉灵龛",
"依栖目那须",
"侯赛因",
"便条",
"俊明",
"保本",
"保琳",
"信博",
"信盛",
"修平",
"修斯",
"修永",
"借来的广告板",
"债务处理人",
"健三郎",
"健司",
"健次郎",
"健谈的村民",
"偷听",
"偷懒的安杰洛",
"傅三儿",
"元助",
"元太",
"元成",
"元能构装体",
"元良",
"元蓉",
"元青",
"元鸿",
"克列门特",
"克罗丽丝",
"克罗索",
"克里托夫",
"克雷门斯",
"入口大门",
"八木",
"公俊",
"公子",
"公明",
"公示",
"兰巴德",
"兰溪",
"兰罗摩和兰加惟",
"兰达",
"兰道尔",
# "兰那罗A",
# "兰那罗B",
# "兰那罗C",
"兰那罗",
"兰那罗的家",
"兴修",
"兴叔",
"兽境猎犬",
"内尔森",
"内村",
"冈崎寅卫门",
"冈崎陆斗",
"冈林",
"冒失的兰那罗",
"冒失的帕拉德",
"冒险家小姐",
"写有缭乱字迹的叶子",
"写有认真笔迹的叶子",
"冰封之心",
"凉子",
"凝光的瓷器藏品",
"几原凯",
"凯西娅",
"凶恶的男人",
"刀疤刘",
"判官儿",
"利彦",
"制作便当",
"刻痕",
"削月筑阳真君",
"前田",
"剑道家",
"剑鱼二番队代理队长",
"剑鱼二番队老兵",
"加埃特",
"加斯帕尔",
"加斯顿",
"加藤信悟",
"加藤洋平",
"勘定奉行足轻",
"勘探队营地告示牌",
"勘解由小路健三郎",
"勿忘",
"包巴卜",
"匆忙写就的笔记",
"北村",
"千岩军",
"千岩军兵士",
"千岩军士兵",
"千岩军守卫",
"千岩军教头",
"千晶",
"半四郎",
"华清",
"卓也",
"卖力的西瑞",
"博易",
"卡丁",
"卡乌斯",
"卡兰塔里",
"卡卡塔",
"卡塔琳娜",
"卡瓦贾",
"卡西姆",
"卡西特",
"卡门",
"卡齐姆",
"卢卡",
"卢妮雅",
"卢森巴博",
"卢正",
"卫兵",
"参赛者",
"又",
"友人",
"友浩",
"友里",
"反复删改的留信",
"发黑的糖罐",
"受伤的狼",
"受影响的须弥蔷薇",
"变魔术的男性",
"古山",
"古拉姆",
"古旧的记事本",
"古旧的路牌",
"古泽",
"古田",
"古老的命令书",
"古老的石碑",
"古老的碑文",
"古谷升",
"叩访仙府",
"可疑的市民",
"可疑的物品",
"可疑的路人",
"叶夫卡",
"叶戈尔",
"叶菲·雪奈茨维奇",
"合成台",
"吉川",
"吉祥",
"同人摊位公告板",
"向香炉进香",
"君君",
# "吟游诗人B",
# "吟游诗人C",
"吟游诗人",
"吴老七",
"吴老二",
"吴老五",
"告示牌上的留言",
"周平",
"周良",
"周顺儿",
"和昭",
"咚咚小圆帽",
"咲耶",
"哈夫丹",
"哈比",
"哈特曼",
"哈里",
"哈里森",
"哈马沙",
"哈鲁特",
"响太",
"哲伯莱勒",
"哲夫",
"哲平",
"商人",
"商人信众",
"商华",
"商家的告示牌",
"商家的海报",
"商店顾客",
"喜儿",
"嘉久",
"嘉信",
"嘟嘟通讯仪",
"回忆中的小五",
"回忆中的木木",
"回忆中的狼哥",
"回忆中的老孟",
"回忆中的聪子",
"图马特",
"土门",
"坂本",
"埃勒曼",
"埃尔欣根",
"埃德蒙德",
"埃舍尔",
"城门守卫",
"堆放",
"塔什芬",
"塔娜玛尔",
"塔尼娜",
"塔德菈",
"塔杰·拉德卡尼",
"塔米米",
"塔里克",
"塞娜",
"塞琉斯",
"墨染像",
"壁画",
"多惠",
"夜兰的太爷",
"夜枭雕像",
"夜航员",
"夜鸦团长",
"夜鸦船长",
"大久保三左卫门",
"大仓",
"大伴",
"大助",
"大姐头",
"大岛",
"大徐",
"大慈树王",
"大森",
"大猎犬",
"大辅",
"大门",
"大隆",
"大雨",
"大黄",
"天叔",
"天守阁守卫",
"天平",
"天成",
"天权幕僚",
"天目优也",
"天目十五",
"天音",
"天领奉行士兵",
"天领奉行护卫",
"失落的市民",
"失页的笔记",
"奇妙的船",
"奇怪的花瓶",
"奇怪的记事本",
"奇怪的雕像",
"奇恩",
"奏太",
"奔奔",
"奔雷手秦师傅",
"奥兹",
"奥列格",
"奥尔罕",
"奥泰巴",
"奥米德",
"女士",
"女性信众",
"好奇的子瑞",
"好奇的帕琪",
"好奇的村民",
"如意",
"妮基娜",
"妮露法",
"娜德瓦",
"娜斯米儿",
"娜赞",
"婕叶",
"婵儿",
"婷婷",
"嫣朵拉",
"字迹不一的笔记",
"字迹华丽的信",
"字迹工整的留信",
"字迹歪歪扭扭的记事",
"字迹潦草的笔记",
"字迹端正的笔记",
"孝利",
"孟丹",
"孟迪尔",
"季同",
"学徒",
# "学者女",
# "学者男",
"学者",
"孩童",
"宇奈",
"宇陀",
"安全告示",
"安妮塞",
"安娜斯塔西娅",
"安彦太郎",
"安抚驮兽",
"安普叔",
"安武",
"安纳亚",
"安贞",
"安迦",
"安顺",
"宏飞",
"宛烟",
# "实验者A",
# "实验者B",
# "实验者C",
# "实验者D",
"实验者",
"审问",
"宫地",
"害怕的玛伽",
"害怕的赫尔斯",
"家丁",
"宽则",
"寂寞的小朋友",
"密兹里",
"富拉特",
"寒锋",
"寡言的廷方",
"寻人告示板",
"寻犬启事",
"封藏资料「归寂之庭」",
"将司",
"小乌维",
"小乐",
"小五",
"小伟",
"小六",
"小冥",
"小千",
"小吉",
"小威",
"小川",
"小德",
"小心！",
"小斌",
"小柳",
"小漫",
"小狗",
"小猫",
"小王子",
"小玫瑰",
"小玲",
"小畑",
"小白",
"小百合",
"小茂",
"小野寺",
"小黑猫",
"小龙",
"尖刻的盗宝团成员",
"尚",
"尤苏波夫",
"尼古拉下士的信",
"尼古拉下士的日志残片",
"尼古拉下士的笔记",
"尼特",
"层岩巨渊爆破组通知",
"居安",
"山本",
"山田一二三",
"岑二",
"岑大",
"岩夫",
# "岩田(废弃）",
"岩田",
"岩藏光造",
"岳川",
"岻伽",
"巡林犬",
"巡逻足轻",
"工作手册",
"工整的字条",
"巫女",
"巴丝玛",
"巴克尔",
"巴哈利",
"巴子",
"巴登",
"巴迪斯",
"布伦",
"布祖里格",
"希拉非",
"希琳",
"帕丽莎",
"帕奇",
"帕慕克",
"帕特里克",
"带有谜语的题板",
"常丰",
"幕府军人",
"幕府足轻",
"幕府随从",
# "平井（废弃）",
"平井",
"平山",
"平泉",
"年迈学者",
"幸也",
"幸德",
"幸雄",
# "幻想朋友刀疤脸斯坦利",
# "幻想朋友琴",
# "幻想朋友诺拉",
# "幻想朋友遗迹守卫",
"幽夜菲谢尔",
"广告",
"广竹",
"庆次郎",
"库什基",
"库因汀",
"库塔",
"库达里",
"应急补给点",
"康介",
"康拉德",
"康明",
"康胜",
"开始绘画",
"开始调制吧",
"开始调酒",
"异人龛",
"弗谢沃洛德",
"弗里茨",
"归终机维护须知",
"彦博的字条",
"彩香",
"影",
"影狼丸",
"彼得",
"往日的黑蛇骑士",
"征文告示",
"御琉部栖",
"御舆长正",
"御陵墓石",
"德利瓦",
"德富",
"德拉夫什",
"德田",
"德鲁苏",
"忍犬",
"志成",
"志琼",
"志穗",
"志织",
"忠夫",
"忠胜",
"思勤",
"思鹤",
"急躁的瓦里特",
"怪异的石碑",
"怪鸟",
"恒雄",
"恕筠",
"恩古尔",
"恩忒卡",
"悉度",
"悠也",
"悠策",
"悦",
# "愚人众",
# "愚人众交易人员",
# "愚人众先遣队",
# "愚人众士兵",
# "愚人众士官",
# "愚人众守卫5",
# "愚人众守卫6",
# "愚人众新兵",
"愚人众的日志本",
"愚人众笔记",
"愚人众行动日志·其一",
"愚人众行动日志·其三",
"愚人众行动日志·其二",
"愚人众行动日志·其四",
"慕珍",
"懒散的小猫",
"戈布利",
"戎世",
"成濑",
"戚楠",
"戴因斯雷布",
"戴派",
"手岛",
"扎哈尔",
"托克",
"扶柳",
"拆封的书信",
"拉伊德",
"拉凯什",
"拉多米尔",
"拉娜",
"拉巴哈",
"拉德普",
"拉扎克",
"拉特什",
"拉齐",
"拓真",
"拜德尔",
"掇星攫辰天君",
"接触镇物",
"提取装置",
"提尔扎德",
"提莫尔",
"支支",
"收集材料的兰那罗",
"收集线索",
"敌方蕈兽",
"教令院卫兵",
"教令院卫队长",
"教令院学者",
"教令院护卫",
"教令院考察队临时报告",
"教令院调查团的报告",
"散兵",
"散兵机甲",
"敲一敲",
"敲窗",
"敲门",
"整理信件",
"整齐堆放的卷轴",
"文泽",
"文渊",
"文璟",
"断刀冢",
"斯内库",
"斯托尼",
"斯格鲁奇",
"斯里玛蒂",
# "新生学者丙",
# "新生学者乙",
# "新生学者甲",
"新生学者",
"新鲜的莲蓬",
"施工注意！",
"旁白",
"无名",
"无怨",
"日志",
"日志的残片",
"旧库房安置通知",
"旧花瓶",
"早期合影",
"昆钧",
"昌信",
"昌贵",
"明俊",
"明博",
"明博的规划书",
"明星斋海报",
"明石",
"明蕴镇告示牌",
"春水",
"晋优",
"晓东",
"晓飞",
"景明",
"景澄",
"暝彩鸟",
"暮夜剧团团长",
"暴徒丘丘人",
"曜",
"更改盆景布局",
"曼达娜",
"月城",
"有泽",
"有香",
"朋义",
"朔次郎",
"望月",
"望雅",
"朝南",
"木下",
"木奈",
"木户",
"木木",
"木村",
"木鲁瓦",
"未来",
"本",
"札齐",
"朱巴",
"朱特",
"朱达尔",
"朱陶",
"机关装置",
"机械残骸",
"机械生命研究资料",
"杉本",
"村山",
"村田",
"杜拉夫",
"杞平",
"杨尼斯",
"杰夫",
"杰巴里",
"杰拉德",
"杰萨尔",
"杰赫南",
"松坂",
"松浦",
"松鼠",
"林逃山",
"枫原久通",
"枫原义庆",
# "枫原义庆-无斗笠版",
"枫原景春",
"柊千里",
"柊慎介",
"查看",
"查看愚人众的研究",
"查看病患",
"柳叶岚士",
"柳达希卡·雪奈茨芙娜",
"柴助",
"柴染",
"柴毅",
"标识牌",
"栖令比御",
"桂一",
"桂木",
"桃子",
"桥本",
"桦山",
"梦境海螺",
"梨香",
"梵米尔",
"森彦",
"椅子",
"楠楠",
"榎本",
"模样怪异的士兵",
"歌特琳德",
"正人",
"正胜",
"正茂",
"武士",
"武沛",
"残存的记录·一",
"残存的记录·三",
"残存的记录·二",
"残旧的记事",
"残破的出勤记录",
"残破的刻字",
"残破的手记",
"残缺的文字",
"殿中监指示",
"毁损的遗迹守卫",
"毓秀",
"比螺梦门",
"民众",
"水史莱姆",
"水手",
"水池",
"水没遗迹钟声",
"水龙蜥",
"永业",
"永贵",
"永野",
"江木",
"池田总四郎",
"汲水",
"沙坎",
"沙姆斯",
"沙寅",
"沙拉夫",
"没写完的补给日志",
"治一郎",
"治安公告",
"法伊兹",
"法伯德",
"法卢克",
"法尔希",
"法尔扎",
"法尔罗赫",
"法拉比",
"波林",
"注意！",
"注意！！",
"泰久",
"泰勒",
"泽田",
"洛伽",
"洛成",
"活跃的欧琳",
"派安",
"派斯利",
"流寇头目",
"浅川",
"浅野",
"浇水",
"浩司",
"浪人",
"浪人暗部",
"浮舍",
"海亚姆的病历表",
"海伯",
"海宁",
"海斗",
"海芭夏",
"海达",
"涂涂改改的盗宝团笔记",
"涉川",
"淘气的兰那罗",
"深渊使徒",
"深渊司铎",
"深渊法师",
"深见",
"清人",
"清刚",
"清姨",
"清惠",
"清水",
"渊上",
"渡部",
"温柔的声音",
"源琴美",
"溜溜",
"漂流瓶",
"演员怪",
"潦草的笔记",
"潮哥",
"澜阳",
"火一",
"火炽之心",
"灯谜",
"灰灰",
"热恋中的少女",
"热恋中的青年",
"焦躁的市民",
"爱德琳",
"爱拉尼",
"爱音乐的兰那罗",
"牛志",
"牧梨",
"物灵",
"特洛芬·雪奈茨维奇",
"特瓦林",
"犬少将",
"狼哥",
"猫",
"猫咪",
"玛亚姆",
"玛克林",
"玛拉",
"玛鲁特",
"玥文",
"玥辉",
"玲花",
"玻瑞亚斯",
"珐露珊母亲",
"珠函",
"班卡",
"班达克",
"理正",
"琬玉",
"琳",
"琳琅",
"琳阳",
"琴美",
"瑛太",
"瑞娜",
"瑞锦",
# "璃月嘉宾A",
# "璃月嘉宾B",
"璃月嘉宾",
"瓦京",
"瓦尔特",
"生病的狗狗",
"用高",
"田中",
"由真",
"甲斐田龙马",
"男性信众",
"畑中",
"留言",
"留言板",
"疲惫的村民",
"病历记录",
"百代",
"百雷遮罗",
"皮特",
"盈丰",
"盈珠",
"盖曼",
"盗宝团",
"盗宝团书信",
"盗宝团小弟",
"盗宝团成员",
"盗宝团日志",
"盗宝团的笔记",
"盗宝团老大",
"盗宝鼬训练手册",
"看似理性的女性",
"看似理性的男性",
"真",
"真昼",
"真理",
"眼神凛冽的夜鸦",
"矢部",
"矢野町子",
"知世",
"知易",
"知易的规划书",
"知论派学者",
"石原",
"石崎",
"石川八郎",
"石田",
"石镇子",
"矿工",
"矿工宿舍紧急通知",
"码头的工人",
"破旧的广告牌",
"破破烂烂的盗宝团笔记",
"礼安",
"祈愿的篝火",
"祝明",
"神奇的霍普金斯",
"神情严肃的夜鸦",
"神秘的信笺",
"神秘的女性",
"神秘的留言",
"神秘的障壁",
"神秘的雕像",
"神里家主",
"神龛",
"祥生",
"祭拜的女性",
"祭祀指南",
"禁闭室",
"离可琉",
"秀夫",
"秀秋",
"秋人",
"秘密文献解读资料",
"稻生",
"穆妮尔",
"穆尔塔达·拉德卡尼",
"穆泰尔",
"空",
"端雅之墓",
"竹内",
"竹叔",
"竹篮",
"竹谷",
"笔迹已经模糊的笔记",
"笔迹潦草的盗宝团笔记",
"笔迹潦草的笔记",
"笔迹端正的日记",
"符景",
"笼钓瓶一心",
"简朴的墓碑",
"米农",
"米尔",
"米尔萨德",
"粗糙的涂鸦",
"系统音",
"素达蓓",
"索林",
"索赫蕾的笔记",
"紫微",
"约纳斯",
"纪念石",
"纪芳",
"纪香",
"纯也",
"纳吉布",
"纳巴提",
"纳比尔",
"纳瓦兹",
"纳西尔",
"纳赛尔",
"纸张泛黄的留信",
"纸条",
"绀田传助的笔记",
"终末番忍者",
"绍元之墓",
"经纶",
"结菜",
"绘真",
"维修",
"维拉夫",
"维斯科",
"维沙瓦",
"绿色的家伙",
"缄封的命令书",
"罗伊斯",
"罗娅",
"罗尔德",
"罗巧",
"罗杰",
"罗盘的声音",
"罗福",
"罗纳克",
"美和",
"美羽",
"美铃",
"翠光像",
"翰学",
"老仆",
"老何",
"老克",
"老吴",
"老墨",
"老孟",
"老戴",
"老板",
"老蔡",
"老贾",
"老郑",
"老黑七",
"耕一",
"聆听",
"聪",
"聪子",
"聪美",
"胜家",
"胡塞尼",
"胭儿",
# "胶囊npc小孩",
"自称「唐无仇」者",
"自称「渊上」之物",
"舍利夫",
"舒伯特",
"舒杨",
"舒茨",
"般底",
"船夫",
"良子",
"良平",
"艾依曼",
"艾德娜",
"艾斯特尔",
"艾米尔",
"艾莉丝",
"艾莎",
"艾莲娜",
"艾贝",
"芊芊",
"芙萝拉",
"芬尼克",
"花散里",
"芷巧",
"苏尔塔尼",
"苏珊",
"苏蒂娜",
"苏醒的病患",
"若山小十郎",
"若山敬助",
"苦恼的丘丘人",
"苦恼的特利芙",
"范木堂主",
"范火头的配方",
"茅葺一庆小说样稿",
"茱萸",
"茶博士刘苏",
"荒川幸次",
"荣江",
"荧",
"荷贝特",
"莉尔",
"莎迪耶",
"莫塞伊思",
"莫娜的实验室",
"莫孜甘",
"莫尔吉",
"莱妮",
"莱拉",
"莱昂",
"莱蒂法",
"莱诺",
"菜菜子",
"菱田",
"菲利特",
"菲恩",
"菲谢尔的母亲",
"菲谢尔的父亲",
"萍姥姥",
"萨勒",
"萨古",
"萨梅尔",
"萨福万",
"落灰的餐具",
"葵之翁像",
"蒂雅",
"蒙德百货订货板",
"蒙蒙",
"蒲泽",
"藤木",
"藤田",
"蛎罗",
"蛮横的武士",
"行囊与书籍",
"被困居民",
"装有谜语的题箱",
"裕子",
"裕明",
"西乡",
"西拉杰",
"西敏",
"西达",
"西风骑士",
"观察伤兵",
"观察发簪",
"观察图案",
"观察塞西莉亚花",
"观察夜泊石",
"观察宣传单",
"观察小灯草",
"观察山洞",
"观察帐篷",
"观察房门",
"观察文具",
"观察木板",
"观察杯子",
"观察桌子",
"观察椅子",
"观察烹饪的痕迹",
"观察物资",
"观察甜甜花",
"观察画架",
"观察痕迹",
"观察盐化信徒",
"观察盐花",
"观察石块",
"观察石椅",
"观察置物架",
"观察脚印",
"观察花瓶",
"观察茶杯",
"观察蒲公英",
"观察装置",
"观察记录板",
"观察酒杯",
"观察钩钩果",
"观察铃铛",
"观察锅",
"观察雕像",
"观察风筝",
"观察风车菊",
"观察香炉",
"观察香膏",
"观察黑主画像",
"解剖记录",
"解翠行告示牌",
"触摸神像",
"触碰",
"触碰壶面",
"警告",
"记事",
"诊断报告",
"诗筠",
"诗羽",
"诡异的烟雾",
"诺曼",
"诺艾尔的学习笔记",
"课程小结",
"调整火候",
"调查地图",
"调查大炮",
"调查岩王帝君尸体",
"谜样的人影",
"谜样的男性",
"谢夫凯特",
"谢尔盖",
"豆助",
"豆豆",
"贝儿",
"贝琳达",
"贝莲",
"财进",
"货车",
"贵安",
"费尔比",
"贺摩",
"贾瓦德",
"贾维",
"赛诺部下的佣兵",
"赞玛兰",
"赞迪克的笔记",
"赤人像",
"赫尔曼",
"赫里斯托",
"赵铁牛",
"赵铁牛的账本",
"足青",
# "路人npc女",
# "路人npc女02",
"路牌",
"转转悠悠兽",
"轰大叔",
"轻策之藏大门",
"轻策之藏祭坛",
"较新的告示",
"较老的告示",
"辉山厅警示牌",
"辛焱的母亲",
"辛焱的父亲",
"辛秀",
"辛程",
"达丽娜",
"达列尔",
"达尼拉",
"达纳",
"达莉亚",
"达诺",
"迦雅",
"迪吉",
"迪娜泽黛",
"迷茫的兰那罗",
"迷路的小猫",
"通知",
"逢岩",
"逸轩",
"遇袭的小孩",
"遗落的笔记",
"遗迹守卫",
"遗迹宝藏",
"遗迹猎者",
"遗迹的铭文",
"遗迹石碑",
"遗迹考察日志",
"那先朱那",
"酒井",
"醉今朝",
"采珊",
"采集",
"释放人质",
"里栖太御须",
"野伏众",
"野崎",
"野方",
"野武士",
"野生水蕈兽",
"金忽律",
"铁块儿",
"铁算盘",
"铜雀",
"铭文",
"锅巴",
"锋",
"锦野玲玲",
# "镀金旅团C",
"镀金旅团",
"镀金旅团成员",
"长冈秀满",
"长平",
"长次",
"长濑",
"门扉",
"阅读信件",
"阅读日常恋爱故事…",
"阅读璃月武侠故事…",
"阅读转生冒险故事…",
"队长",
"阳太","阳太","阳太","阳太","阳太",
"阳斗",
"阿万",
"阿丑",
"阿久",
"阿乎剌",
"阿亮",
"阿什克",
"阿什帕齐",
"阿仁",
"阿佐",
"阿倍良久",
"阿兹拉",
"阿内",
"阿创",
"阿利娅",
"阿加什",
"阿加菲娅",
"阿升",
"阿卜",
"阿吉",
"阿塔",
"阿外",
"阿多尼斯",
"阿夫辛",
"阿娜耶",
"阿孜米",
"阿守",
"阿尔伯",
"阿尼斯",
"阿巴克",
"阿巴图伊",
"阿布丁",
"阿布多",
"阿弗拉图",
"阿强",
"阿德勒",
"阿扎尔",
"阿扎木",
"阿拉夫",
"阿提亚",
"阿晃",
"阿洛瓦",
"阿特拉",
"阿玛尔",
"阿瑠",
"阿祇",
"阿米",
"阿米尔",
"阿肥",
"阿莎",
"阿莲佐",
"阿萨里格",
"阿蒙苏",
"阿虎",
"阿诺德",
"阿豪",
"阿贤",
"阿部",
"阿阳",
"阿顺",
"阿鸠",
"阿鸿",
"阿麦努",
"阿龙",
"陈娇花",
"陈旧的手记",
"陈旧的笔记",
"陶义隆",
"陷入爱河的女性",
"随缘的铃官",
"隼人",
"雅普",
"雅科夫",
"雇主的命令",
"雕像",
"雪地丘丘人",
"雷萤术士",
"青木",
"青白",
"青莲",
"韦尔纳",
"音上",
"音乃",
"韵宁",
"顺平",
"须婆达之彦",
"须弥考察队的笔记",
"风声",
"风魔龙",
"风龙",
"飞飞",
"香川",
"香炉",
"马克利",
"马坎",
"马姆杜",
"驮兽",
"驹形",
"高善",
"高坂",
"高山",
"高志",
"高级学者",
"高老六",
"鬼婆婆",
"魏风尘",
"鲁哈维",
"鲁希",
"鲁达贝",
"鲍连卡的「指令」",
"鲍里斯",
"鲱鱼一番队队员",
"鲸井小弟",
"鸿兴",
"鹫津",
"鹰司千歌",
"鹰司家臣",
"鹰司进",
"鹿庭一庆",
"麦娜尔",
"麦希尔",
"麻美",
"黄衫",
"黄麻子",
"黑主像",
"黑谷势至丸",
"黛比",
"齐里",

# other leak from huiyadanli
# "#{NICKNAME}",
# "#{REALNAME[ID(1)|HOSTONLY(true)]}",
# "(Test)平原料理任务NPC",
# "(Test)捕鱼挑战NPC",
# "(Test)星火",
# "(Test)桥西",
# "(Test)漂流瓶收集者",
# "(Test)索拉雅",
# "(Test)车夫",
# "(test）拳手一号",
# "(test）盗宝团电",
# "(test）镜头",
# "???",
# "NPC动作测试1-男",
# "NPC动作测试2-女",
# "NPC动作测试3-女",
# "PhatomSamurai",
# "test伪装的千岩军1",
# "test伪装的千岩军2",
# "test望舒客栈的搬运工",
"「丑角」",
"「仆人」",
"「假编剧」",
"「偷懒鬼」",
"「全勤王」",
"「公主」",
"「公子」",
"「勇士」",
"「卡布斯」",
"「呜呜大葡萄」",
"「呜呜葡萄」",
"「大冒失」",
"「女士」",
"「小水珠」",
"「工程师」",
"「得到回复的假条」",
"「总导演」",
"「散兵」",
"「时髦墩墩桃」",
"「晶螺糕站」安全操作规范",
"「朴实钩钩果」",
"「残酷嘟嘟莲」",
"「没疤脸」帕奇诺",
"「法拉西亚」",
"「狗狗」",
"「狡诈泡泡桔」",
"「王子」",
"「玛丽安」",
"「白老先生」",
"「管饭的」",
"「粉刷匠」德尼禄",
"「纳西索斯·四天王之一」",
"「纳西索斯的邪恶爪牙」",
"「老板」",
# "「胆怯」的守卫4",
"「胆怯」的守卫",
"「西摩尔」",
"「贡达法」",
# "「贪婪」的守卫3",
"「贪婪」的守卫",
"「那维莱特的假条」",
"「邪恶爪牙·三人众之一」",
"「金龟奇术番长」",
"「集群」成员",
"一个故事",
# "七七线盗宝团1",
# "七七线盗宝团2",
"不太礼貌的学舌鸟",
# "丘丘人1",
# "丘丘人2",
# "丘丘人3",
"丹纳",
"丽什蒂",
"乌宰尔",
"乌布",
"乌格威",
"乌迪诺",
"乌鸦",
# "乌鸦1",
# "乌鸦2",
# "乌鸦3",
"乐园告示喇叭",
"于托先生",
"于贝尔",
"人群",
"仇敌的伏兵",
"伊廖沙",
"伊德里西",
"伊斯梅诺",
"伊梅娜",
"伊洁贝儿",
"伊洛丝",
"伊莉米娅",
"伊薇特",
"伊诺米娅",
"伊迪娅",
"伊雅蕾",
"众水手",
"伦纳德",
"伯德里科",
"伽吠毗陀",
"佐西摩斯",
"佩尔索内",
"依维莱琳",
"修复奔奔",
"光之",
"克兰茨",
"克叙涅",
"克洛妮艾",
"公告牌",
"兰丘",
"兰浮婆",
"兰莎妮",
"兰道夫",
# "兰那罗A",
# "兰那罗B",
# "兰那罗C",
"兴奋的观众",
"兴晔",
"冬雅",
"决定上午的安排",
"凯伊卡",
"凯库巴德",
"凯特上校",
"凹陷的痕迹",
"刁民",
# "刁民3",
"利瓦尔",
"利露帕尔",
"前往「特别温暖的地方」",
"加特尼奥",
"加里克",
# "动作测试用派蒙1",
"劳莉",
"勒克莱德",
"勒非尼",
"匆忙的囚犯",
"千世",
"博纳",
"卡万",
"卡兰萨",
"卡利贝尔",
"卡特皮拉",
"卡莉娜",
"卡莉珀丝",
"卡莉露",
"卡萝蕾",
"卡西米",
"卢蒂妮",
"厄代尔",
"厄里那斯",
"古拉卜",
"古老的博物志节选",
"古老的记事",
"古齐安",
"可疑的木箱",
"可疑的男人",
"叶夫格拉夫",
"叹气的观众",
"吉他手",
"吉约丹",
"吉雅罗",
# "吟游诗人B",
# "吟游诗人C",
"吟游诗人",
"命运使徒",
# "和谈现场的反抗军士兵1",
# "和谈现场的反抗军士兵2",
# "和谈现场的反抗军士兵3",
# "和谈现场的反抗军士兵4",
# "和谈现场的反抗军士兵5",
# "和谈现场的幕府士兵1",
# "和谈现场的幕府士兵2",
# "和谈现场的幕府士兵3",
# "和谈现场的幕府士兵4",
"哈亚提",
"哈伦",
"哈多",
"哈姆宰",
"哈彦",
"哈拉夫",
"哈比卜",
"哲俩阿",
"哲瓦德",
"喀什米",
"善意的提示",
"嘉玛",
"嘉维娜",
"嘉良",
"嘉莱娜",
"嘲笑的观众",
"团雀",
"困惑的观众",
"图斯",
"图昂",
"坩埚顶部",
"埃克朗谢",
"埃尔瓦德",
"埃德",
"埃德蒙多",
"埃桑",
"埃纳夫",
"基娅拉",
"基肖恩",
"塔尼特部族成员",
"塔尼莎",
"塔耶芙尔",
"塞勒",
"塞卡伊",
"塞律里埃",
"塞萨尔的日记",
"多梅尼科",
"夜间守卫",
"大副",
"大卫",
"大赛公告",
"大雷",
# "天守阁守卫 倒地",
"奇怪的人",
"奇怪的发条机关",
"奇迹建筑师",
"奈兰",
"奥列斯特",
"奥利维亚",
"奥瑟莱",
"奥蒂涅",
"妮欧莎",
"威尔船长",
"字迹稚嫩的日志",
"学生",
"守望者的长老",
"安",
"安里",
# "对战的反抗军1",
# "对战的反抗军2",
# "对战的幕府军1",
# "对战的幕府军2",
"对话",
"寻找",
"小卷心菜",
"小郁",
"尚博迪克",
"尚帕隆",
"居勒什",
"巡林员",
"巡林护卫",
"工程师的笔记",
"巴哈多",
"巴哈尔",
"巴塔索",
"巴朗",
"巴沙尔",
"巴特洛",
"巴达维",
"巴里亚",
"布兰奇",
"布吕诺",
"布尔米耶",
"布瓦列特",
"布瓦洛",
"布罗意",
"布里科勒",
"希塞蕾",
"希沙姆",
"希洛娜",
"希茉妮",
"希露艾",
"帕兹",
"帕巴格",
"帕帕克",
# "幻影路人A",
# "幻影路人B",
# "幻影路人C",
# "幻影路人D",
"幻想朋友刀疤脸斯坦利",
"幻想朋友琴",
"幻想朋友诺拉",
"幻想朋友遗迹守卫",
"广播",
"库塞拉",
"库洛德莫",
"库特罗",
"库珀",
"廷稚",
"开启档案室",
"开朗的男人",
"弈丰",
"弗莱谢",
"张大姐",
"德尔菲娜",
"德尼谢尔",
"德帕里",
"德拉索",
"德拉萝诗",
"德拉萝诗的笔记",
"德拉萝诗的鱼钩",
"德瑟布尔",
"德皮耶里",
"思考的观众",
"恐慌的佣兵",
"恩肖",
"惊讶的观众",
"愉悦的幽魂头领",
"意犹未尽的观众",
"愚人众",
"愚人众交易人员",
"愚人众先遣队",
# "愚人众先遣队员乙",
# "愚人众先遣队员甲",
"愚人众士兵",
"愚人众士官",
# "愚人众守卫5",
# "愚人众守卫6",
"愚人众守卫",
"愚人众新兵",
"感慨的观众",
"愤怒的幽魂头领",
"愿望清单",
"戈尔代",
"戈莉",
"扎德拉",
"托皮娅",
# "托马coop围观npc1",
# "托马coop围观npc2",
# "托马coop围观npc3",
# "托马coop围观npc4",
# "托马coop围观npc5",
# "托马coop学员女1",
# "托马coop学员女2",
# "托马coop学员女3",
# "托马coop学员男1",
# "托马coop学员男2",
# "托马coop清扫挑战npc1",
# "托马coop清扫挑战npc2",
"拉库马尔",
"拉斯蒂涅",
"拉杜",
"拉沙鲁",
"拜希麦",
"接待员",
"提克里蒂",
"提尔贝特",
"提普",
"提示器",
"摩婕娜",
"散落的书页",
"散落的纸",
"敲敲箱子",
"斯嘉莉",
"斯露莎",
"新浪潮文汇·上册",
"新浪潮文汇·下册",
"新浪潮文汇·中册",
"方妮雅",
"施芮娅",
"旋转火蕈兽",
# "昆恩（事件测试）",
"普吕姆",
"普扬",
"普莱希雅",
"曼多拉",
"朱利亚诺",
"机械狗",
"杜吉耶",
"来自喇叭的旁白",
"来自稻妻的长途信件",
"来自须弥境内的信件",
"杰娜姬",
"杰柯艾妲",
"杰洛尼",
# "枫原义庆-无斗笠版",
"某人",
"查看水面",
"查阅轨道线路",
"柯莉黛尔",
"柯莎",
"格内薇芙",
"格拉西亚",
"格罗斯利",
"格莱",
"格蕾丝蒂",
"格蕾欣",
"桑尼",
"桑迪诺",
"梅尔克",
"梅瑟娜",
"梅菈",
"梅赫拉克",
"欧莉艾尔",
"正义机器",
"比勒",
"毕洛",
# "气泡A",
# "气泡B",
# "气泡C",
# "气泡D",
# "气泡E",
"水形幻灵",
"江蓠",
"沃特林",
"沉思的观众",
"沙尔梅",
"沙扎曼",
"沙狐",
"沙蒂永",
"沙赫布特",
"法乌菈",
"法图赫",
"法尔博",
"法拉娜",
"法莉巴",
"法赫尔",
"波蒂埃",
"泽娜",
"泽彦",
"洛伦佐",
"洛梅",
"洛泰尔",
"洛耶茨",
"浮游水蕈兽",
"浮游水蕈兽·元素生命",
"浮游风蕈兽·元素生命",
"海鸥",
"演出公告",
"激动的观众",
"热塞尔",
"熙德拉",
"片场备注",
"犹豫的佣兵",
"狗",
"猛烈纯洁坩埚",
"猛烈纯洁小屋",
"玛卡德尔",
"玛尔瓦",
"玛梅赫",
"玛梅赫的小屋",
"瑟法娜",
"瑟琪",
"瑟琳",
"瑟米安",
"瑟维妮",
# "璃月嘉宾A",
# "璃月嘉宾B",
"瓦伊兹",
"瓦瑟迪瓦",
"瓦纳格姆",
"瓦谢",
"瓶子",
"甘珠尔",
"画布",
"疑惑的观众",
"疲惫的男人",
"皮塔尔",
"皮雅",
"看守",
"看起来不太正经的人",
"看起来似乎有点坏的人",
"看起来稍微有点危险的人",
"研察终端",
# "社奉行武士A",
# "社奉行武士B",
# "社奉行武士C",
# "社奉行武士D",
# "祭典氛围女A",
# "祭典氛围女B",
# "祭典氛围女C",
# "祭典氛围男A",
# "祭典氛围男B",
# "祭典氛围男C",
"种下种子",
"科拉莉",
"笔记",
"箱子",
"米图尔",
"米尔扎",
"米希尔",
"米沙勒",
"素海卜",
"索维格莎",
"纯水精灵？",
"纳克",
"纳兰德拉",
"纳库尔",
"纳瓦伊",
"线索提示板",
"结果公示",
"络紫",
"维勒",
"维卡斯",
"维吉尔",
"维纳亚克",
"维耶尔默",
"维莱妲",
"罗珊",
"罗贝尔·乌丹",
"翻阅教材",
"考威尔",
"耶尔米尼",
"肖像",
"肯顿",
"胡瓦",
"胡维斯卡",
"自信的观众",
"舒瓦瑟尔",
"舒莱赫",
"航海日志",
"艾伊丝",
"艾古伊",
"艾尤恩",
"艾德里安",
"艾文",
"艾玛娅",
"艾玛尔",
"艾玛托",
"艾辛多弗",
"芙佳",
"芙洛",
"芬奇",
"花灵",
"莉莉安",
"莉诺尔",
"莫普瓦",
"莫莱妮",
"莫里",
"莱斯格",
"菲多赫",
"菲德兰",
"菲莉吉丝",
"萨塔尔",
"萨德赫",
"萨莎",
"萨赫哈蒂",
"萨齐因",
"落单的水形幻灵",
"蕾卡",
"薇尔妲",
"薇涅尔",
"薇蕾娜妲",
"蜂巢心声",
# "蜂巢心声1",
# "蜂巢心声10",
# "蜂巢心声2",
# "蜂巢心声3",
# "蜂巢心声4",
# "蜂巢心声5",
# "蜂巢心声6",
# "蜂巢心声7",
# "蜂巢心声8",
# "蜂巢心声9",
"表情低落的犯人",
"表情狰狞的犯人",
# "被骂的反抗军士兵A",
# "被骂的反抗军士兵B",
# "西拉杰02",
"西摩尔",
"西瓦尼",
"观众",
"观察倒下的卫兵…",
"观察柴堆",
"观察梁柱",
"解读书页内容",
"詹妮特",
"警卫机关",
"警备队员",
"话剧放送告示塔",
"诺瓦勒斯",
"调查大门",
"调查痕迹",
"调查窗户",
"谜之声",
# "谜之声A",
# "谜之声B",
# "谜之声C",
# "谜之声D",
"谷木的残灵",
"贝西",
"费索勒",
"费迪南德",
"贾尔贾",
"赛事积分",
"赛场司仪",
"赛达菲",
"赞同的观众",
"达尼奥",
"达拉",
"达梅婕",
"达雅",
"迈蒙",
"进入内室",
"进入地道",
"迪福尔",
"迪迪巴巴",
"遗落的研究笔记",
"邱老板",
"郎蒂",
"醒悟的观众",
"里加斯",
"重力仪",
"野猫",
"钱德拉",
# "镀金旅团C",
"镀金旅团刺客",
"镀金旅团哨兵",
"门内的声音",
"阿丹",
"阿佩普",
"阿尔",
"阿尔亚沙",
"阿尔卡米",
"阿尔图",
"阿尔贝",
"阿尔邦",
"阿尤布",
"阿巴奈尔",
"阿方",
"阿木",
"阿比德",
"阿维丝",
"阿莎拉",
"阿莱",
"阿迪勒",
"阿雩",
"阿鲁埃",
"阿黛尔",
"陆行岩本真蕈·元素生命",
"陆行水本真蕈·元素生命",
"雅克",
"雅各布",
"雷兹吉",
"雷汉",
"雷温",
"雷维",
"雷萨哈尼",
"雷蒙多",
"震惊的观众",
"露瓦卜",
"露菲娜",
"颇有年头的任命状",
"颇有年头的日志",
"颇有年头的研究报告",
"颇有年头的笔记",
"颇有年头的记录",
"马哈",
"马尔科佐夫",
"马林诺夫斯基",
"鳄鱼",
"麦凯什",
"麦扎尔",
"黄麻子众人",
# "黄麻子众人1",
# "黄麻子众人2",
"黑帮成员",
"黑板",
"黛依阿",
"龚古尔替身",
# "（事件Test)订餐人",
"（聆听…）",

'卡茜涅尔',
'表彰状',
'莫索的日志（二）',
'布雷克',
'气泡',
# '(test）盗宝团电',
'对战的反抗军',
'托萝莎',
# '托马coop清扫挑战npc1',
# '蜂巢心声10',
'回过神来的观众',
# '蜂巢心声1',
'暴怒的龙蜥',
'勇猛奋斗日记（三）',
'刻有字迹的贝壳（一）',
'纳齐森科鲁兹',
'「调皮鬼」',
'「孩子们」',
'社奉行武士',
'「勇猛落落莓」的同伙',
'哈维沙姆',
# '和谈现场的幕府士兵1',
# '和谈现场的反抗军士兵3',
'受灾的女性居民',
# '幻影路人B',
'旧照片',
# '(Test)桥西',
# '#{NICKNAME}',
'黄麻子众人',
# '蜂巢心声4',
'肖诺',
'欣慰的观众',
'吟游诗人',
'「缇拉」',
'比罗',
'莫索的日志（三）',
'布朗什',
'波洛',
'布勒',
'让克',
'认真的观众',
'气泡',
'黛丝蕾',
'「小喷泉」',
'兴澄',
'「贪婪」的守卫',
'记录',
'博兰德！郎勃罗梭！',
'故事书',
'泉水精灵',
'肖西韦尔',
'埃兰',
# '托马coop学员女3',
# 'NPC动作测试3-女',
# '托马coop学员女2',
# '幻影路人D',
'担忧的观众',
'考察记录',
# '和谈现场的反抗军士兵5',
'「特尔克西」',
'亢奋的观众',
'拉瑟兰',
'舒蕾',
'镀金旅团',
# 'NPC动作测试1-男',
'百愁解酊剂',
'萨宾',
'杜莎',
'乌鸦',
'不满的观众',
# '祭典氛围男C',
# '（事件Test)订餐人',
'「特尔克西」开发日志（一）',
'佩尔曼雅',
'黄麻子众人',
'特别枫达专卖机',
'贝妮蒂',
'认可的观众',
'某人的喊声',
'愚人众守卫',
'对战的幕府军',
'刻有字迹的贝壳（三）',
'莫尔让',
# '托马coop围观npc1',
'丘丘人',
'弗里曼',
# 'test伪装的千岩军1',
# '祭典氛围女C',
# '社奉行武士C',
'留影',
# '动作测试用派蒙1',
'西拉杰',
'刁民',
# '(test）拳手一号',
'刺玫会成员',
'唐·西哈诺',
'社奉行武士',
'尤维尔',
'珀西芙',
'「特尔克西」开发日志（三）',
'通告',
'祭典氛围男',
'谁人的日志',
'谜之声',
# '丘丘人1',
'脏兮兮的空酒瓶堆',
# 'Name',
'勇猛奋斗日记（二）',
# '丘丘人2',
'枫原义庆-无斗笠版',
'无奈的观众',
# '谜之声B',
'混乱的观众',
# '谜之声D',
# '和谈现场的幕府士兵3',
# '蜂巢心声8',
'雅拉德',
'刻有字迹的贝壳（二）',
# 'test望舒客栈的搬运工',
'勇猛奋斗日记（一）',
'埃松',
'韦亚斯',
# '乌鸦3',
# '社奉行武士B',
# '气泡B',
'吉沃尼',
'卡纳',
'罗谢',
'某人的声音',
'「曲线」',
'克拉佩',
# '璃月嘉宾A',
'康妮',
'沉睡的龙蜥',
# '托马coop清扫挑战npc2',
'蒂蕾娜',
'恍然大悟的观众',
# '乌鸦2',
'卢瓦利耶',
# '???',
# '气泡E',
# '愚人众守卫6',
'舒当',
# 'test伪装的千岩军2',
'前来阻挡的愚人众',
'瑞尔维安',
'加希',
# '托马coop学员男2',
# '被骂的反抗军士兵B',
'神秘祭坛',
# '七七线盗宝团2',
# '被骂的反抗军士兵A',
# '蜂巢心声6',
'勒马克',
'小弟',
'阿芙颂',
'怀疑的观众',
# '(Test)车夫',
'醉汉',
'吉尼亚克',
# '对战的反抗军1',
'「胆小鬼」',
'贾拉康',
'打抱不平的观众',
'格伦德·雪奈茨维奇',
'「大个头」',
'梅尔梅',
# '(test）镜头',
'帕坦',
'菲希里耶',
# '蜂巢心声2',
# '蜂巢心声7',
'约莲妮',
# '托马coop围观npc5',
'刻字的石碑',
'破烂的衣服',
'严肃的观众',
# '(Test)平原料理任务NPC',
# '和谈现场的反抗军士兵4',
'维尔芒',
# '托马coop学员女1',
# '#{REALNAME[ID(1)|HOSTONLY(true)]}',
# '兰那罗C',
# '兰那罗A',
'愚人众头目',
'莱斯科·德斯特雷',
# '蜂巢心声9',
'某不知名小弟',
'勇猛奋斗日记',
# '兰那罗B',
'愚人众先遣队员',
'娜娜',
# '祭典氛围男A',
'艾德温·伊斯丁豪斯',
'「勇猛落落莓」',
'莫索的日志（一）',
# '(Test)捕鱼挑战NPC',
'埃夫拉',
'奇怪的箱子',
# 'PhatomSamurai',
# '七七线盗宝团1',
# '谜之声A',
'金格尔',
# '天守阁守卫 倒地',
# 'NPC动作测试2-女',
# '托马coop围观npc3',
'反驳的观众',
'西芒',
# '气泡C',
'劳维克',
# '祭典氛围女B',
# '托马coop围观npc2',
'伊尔梅',
# '幻影路人C',
# '幻影路人A',
'摘录',
# '和谈现场的反抗军士兵2',
# '祭典氛围女A',
'质疑的观众',
'购买「枫达」',
'尚贝兰',
'迷茫的观众',
'奥蕾丽',
'失望的观众',
'所有人',
'大当家',
# '璃月嘉宾B',
# '吟游诗人C',
# '(Test)星火',
# '对战的幕府军1',
'受灾的男性居民',
'愚人众先遣队员甲',
'「特尔克西」开发日志（二）',
# '和谈现场的反抗军士兵1',
'讥讽的观众',
'介绍',
'祖莉亚·德斯特雷',
'红发青年',
'惊慌的盗宝团',
'摆满餐具的餐桌',
# '(Test)索拉雅',
# '和谈现场的幕府士兵2',
'梅洛·郎勃罗梭',
# '蜂巢心声3',
# '昆恩（事件测试）',
'魔女N',
'不解的观众',
# '和谈现场的幕府士兵4',
'维钦佐',
# '托马coop学员男1',
'玛丽安',
'坎佩尔',
'马德莱娜',
'思索的观众',
'致主编的信',
'看似懦弱的红发青年',
'德巴内',
'欢腾之火',
'乐平波琳',
'古旧的发条吊坠盒',
'昂吉安妮',
# '(Test)漂流瓶收集者',
# '托马coop围观npc4',
# '蜂巢心声5',
'佩妮',
'伯拉尔',
'纳蒂亚',
'皮托',
'勇猛奋斗日记（四）',
'迪尔菲',
'丝柯克',
'桐本',
# '「胆怯」的守卫4',

'晶核',
'鳅鳅宝玉',
'发光髓',
'蜥蜴尾巴',
'蝴蝶翅膀',
'天原银莲',
'烈焰花花蕊',
'螃蟹',
'鳗肉',
'冰雾花花朵',

'黛丝蕾父亲',
'瑞安维尔',
'「赤红一杵」',
'科尔特',
'马蒂纳',
'安格拉雷',
'佩皮',
'路德温',
'莱斯利',
'「水晶岛梆梆旋风」',
'开始对战',
'「花羽叶月神机」',
'「金刚身怒面虫王」',
'开始对战',
'伯努瓦·勒鲁瓦',
'阿托斯',
'莫蒂西娅',
'奇怪的痕迹',
'房屋出售信息',
'皮普',
'史诺德格拉斯',
'薇若妮卡',
'巴蒂斯特',
'莫里斯',
'博诺',
'勒泰利埃',
'埃米雷德',
'会场主持人',
'书摊老板',
'买书的顾客',
'「装备碾压型选手」',
'「超级强者的随行厨神」',
'「天崩石裂超级强者」',
'「且听下次复盘」',
'「恐怖猫爪草」',
'「花里胡哨不如一刀」',
'安娜伊',
'毕朗什',
'碧碧',
'柯蕾丽亚',
'开始拍摄',
'科斯唐坦',
'伊莲',
'梅里爱',
'小露米埃尔',
'进入仓库',
'埃莉莎',
'莫里斯',
'《古华手记》',  
'土生',
'云云',
'古老的摹刻',    
'歪歪扭扭的摹刻',
'碑铭',
'仙家笔记·其一', 
'古老的文本·其一',
'《诸武集成校笺·裁雨法》',
'《诸武集成校笺·刺明法》',
'《百术集全》',
'《悬练九剑》',
'《遗珑埠不容错过的三种小吃》',
'《百术集全》',
'《武理歌诀》',
'凌霄',
'毅钧',
'《古华诸贤谱录》',
'明礼',
'木李',
'霸哥',
'霸爷',
'仁三',
'楚伯',
'小梁',
'贾尔库坦',
'琼玖',
'前韵',
'寒泉',
'旋覆',
# '玄机(弃)',
'青夜',
'红鱼',
'玄机',
'香器铭文',
'古老的玉玦',
'古老的文本·其二',
'古老的文本·其三',
'谁人的闲笔',
'谁人的闲笔',
'谁人的闲笔',
'大桔',
'赛金砖',
'青夜',
'红鱼',
'云云',
'椒椒',
'模糊的碑铭',
'残破的碑铭',
'焦黑的笔记',
'《草堂拾遗》',
'古剑铭文',
'调查样品',
'石碑铭文',
'赤璋舆图',
'先民札记·其一',
'先民札记·其二',
'先民札记·其三',
'古老的卷轴',
'字迹潦草的日记',
'连芳',
'阿绿',
'方矩',
'利兴',
'弗里斯兰',
'九贯',
'捷悟',
'却之',
'百疏',
'小旷',
'笃勤',
'万斛',
'轩瞩',
'永赞',
'黄三爷',
'韩迁',
'赫曼逊',
'有规',
'文华',
'仪正',
'洪分',
'罗叔',
'老陆头',
'椒椒',
'石坤',
'莫妍',
'沃特蒙泰涅',
'大黑',
'平善',
'望涓',
'宜清',
'陈八珍',
'丰泰',
'衡听',
'艾葛',
'阿林',
'老丁',
'小丁',
'大盗',
'泊溯',
'孙桡',
'舒山',
'德音',
'若松',
'可慎',
'文四爷',
'礼信',
'知贵',
'迭跃',
'泰锡封',
'婉静',
'斡流',
'嘉明',
'闲云',
'接笏',
'侯章',
'陇舟',
'知贵',
'沙博尼耶',
'遐庆',
'嘉明阿婶',
'叶德',
'嘉明阿伯',
'嘉明阿叔',
'嘉明阿婆',
'嘉明阿公',
'风涟',
'漱玉',
'远黛',
'远黛',
'有些特别的鹤',
'远黛',
'陷入迷惑的小猫',
'开始制作',
'等待海灯节开幕',
'嘉明阿婶',
'嘉明阿伯',
'嘉明阿叔',
'嘉明阿婆',
'嘉明阿公',
'问柳',
'庆继',
'宝泰',
'砥心',
'漱玉',
'桌上的机关',
'绍祖',
'没有署名的信',
'狐狸',
'野鹤',
'清尘',
'清尘',
'远黛',
'浮锦',
'谁人的笔记',
'赤井',
'马里',
'渚春',
'铁豆儿',
'铁明',
'尊尼',
'红鱼',
'灵渊',
'喂食',
'「慢悠悠仙像」',
'黑帮群演',
'铁船儿',
'沃克',
'青夜',
'奇怪的商人',
'坐地虎',
'明镜',
'铁锤儿',
'铁瓜儿',
'毅钧',
# '黑帮群演1',
'盘枝蛇',
'敲敲门',
# '黑帮群演2',
# '黑帮群演3',
'铁扇儿',
'铁门儿',
# '开战站桩NPC1',
# '开战站桩NPC2',
'铁船儿',
# '黑帮出列俘虏1',
# '枫丹出列俘虏2',
'遐庆',
'文寻',
'卷舒',
'枕鸢',
'接笏',
'侯章',
'叶德',
'桌上散落的信件',
'桌上散落的笔记',
'厄舍',
'莉安',
'韦斯',
'维格尔',
'时装周宣传画报',
'进入千织屋',
'艾洛迪',
'吉莲',
'梅里埃',
'吉娜',
'阿莫瑞',
'马洛尼',
'贝努瓦',
'邦妮',
'路边闲聊的人',
'路边闲聊的人',
'路边闲聊的人',
'路边闲聊的人',
'莱提西娅',
'门外的顾客',
'书柜上的纸张',
'本子上的备忘录',
'桌上的笔记',
'科伦汀',
'热情的记者',
'兴奋的记者',
'西里',
'雅各',
'吉拉尔',
'调查笔记',
'调查旧作',
'调查正式剧本',
'调查「原剧本」',
'调查手稿',
'艾洛迪',
# '警卫机关1',
# '警卫机关2',
# '警卫机关3',
'贝努瓦',
'观察人群',
'观察人群',
'观察人群',
'观察人群',
'留云借风真君…？',
'阿泰菲',
'弗雷瑞斯',
'德佑',
'芙里达',
'「雪糕」',
'「团团灰」',
'「阿呆」',
'「兔叽」',
'「大桔骑士」',
'瓦乐瑞娜',
'谢妮',
]

def random_server_leak_name():
    return random.choice(server_leak_names)