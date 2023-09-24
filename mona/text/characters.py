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

    "？？？",

    # 世界任务
    "夏诺蒂拉",
    
    # 每日委托
    "古德温",   # 触不可及的恋人
    '葛罗丽',
    "龟井宗久", # 全能美食队
    "芭尔瓦涅",
    "旭东",
    "朱莉",
    "丘丘人",    # xx交流
    "寝子",     # 黑猫猫
    "秋月",     # 绝对独特的美食
    "翔太",     # 神明啊，回应我吧！
    "麻纪",     # 众人祈祷中…
    "柚子",     # 这本小说…有问题？
    "长谷川",   # 这本小说…有问题？
    "若心",     # 久久望故人
    "石头",     # 点石成…什么
    "刘苏",     # 且听下回分解
    "一成",     # 无底之胃
    "索拉图",   # 「夺宝」小行动
    "阿旭",     # 这本小说真厉害
    "常九爷",   # 这本小说真厉害
    "江雪",     # 独钓江雪
    "害怕的舒特",# 向冬日回归
    "玉霞",     # 趁热食用
    "米拉娜",   # 此路不通？
    "米歇尔",   # 永不停歇的风与米歇尔小姐
    "艾拉·马斯克",  # xx交流
    "正二",     # 箭术示范
    "哈特姆",   # 吞金和蓄财
    "古拉布吉尔",   # 宝贝计划
    "贾法尔",   # 食与学
    "「拉勒」", # 喵…喵喵？喵！喵。（找猫猫）
    "加尔恰",   # 加尔恰的赞歌·关键物品
    "路通",     # 加尔恰的赞歌·关键物品
    "阿汉格尔",  # 加尔恰的赞歌·轴承在上
    "福本",     # 《召唤王》
    "拉玛",     # 货比两价

    # 大世界，不想写爬虫了
    "瑟萝",
    "勒芒德",
    "赫茉莎",
    "薇娜",
    "谢里埃",
    "维缪尔",
    "乔尔",
    "乔瑟夫",
    "亨特",
    "恩里",
    "布列松",
    "伊萨克",
    "东雅",
    "贝海姆",
    "拜伊",
    "醒醒", # 猫
    "布特罗斯",
    "欧欧", # 奥摩斯港猫，猫，猫！
    "塔拉内",
    "宫岛",
    "木南杏奈",
    "优素福",
    "内嘉",
    "库玛莉",
    "斯汪", # 是狗啊是狗啊是狗啊！
    "曼苏尔",
    "伊丽丝",
    "博来",
    "张顺",
    "伊凡诺维奇",
    "乾子",
    "苏二娘",
    "杰里",
    "东升",
    "快刀陈",
    "永安",
    "百闻",
    "百识",
    "百晓",
    "伍德",
    "蓝川丞",
    "松本",
    "甘乐",
    "卯师傅",
    "梅琪妮",   
    "阿尔芒",
    "凯瑟琳",
    "怀尔德",
    "莎拉",
    "优律",
    "巴顿",
    "查尔斯",
    "宁禄",
    "尼尔森",
    "派恩",
    "葛瑞丝",
    "苏西",
    "门罗",
    "萨义德",
    "六指乔瑟",
    "维多利亚",
    "法拉",
    "薇尔",
    "阿拉米",
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
    "莉莉",
    "图雷尔",
    "史蒂文斯",
    "齐格芙丽雅",
    "玛格丽特",
    "诺玛",
    "米哈伊尔",
    "安东",
    "大祐",
    "利文斯通博士",
    "艾迪丝博士",
    "艾伦",
    "苦恼的莉安",
    "胆小的莫罗",
    "惊慌的米拉娜",
    "勒戈夫",
    "铁衫",
    "布洛克", # B吃肉的
    "梅薇丝",
    "山田",
    "葵",
    "阿芬迪", # 须弥声望
    "贾汉",
    "库洛什",
    "稻城萤美",
    "快拳阿凌",
    "霹雳闪雷真君",
    "芬德",

    # 兰那罗 from bwiki spider
    "兰拉娜",
    "兰罗摩",
    "兰拉迦",
    "兰帕卡提",
    "兰茶荼",
    "兰萨卡",
    "兰阿帕斯",
    "兰非拉",
    "兰加惟",
    "兰耶娑",
    "兰修提袈",
    "兰陀娑",
    "兰纳真",
    "兰迦鲁",
    "兰纳迦",
    "兰般度",
    "兰那库拉",
    "兰百梨迦",
    "兰贡迪",
    "兰伽卢",
    "兰利遮",
    "兰犍多",
    "兰伊舍",
    "兰梨娄",
    "兰难世",
    "兰穆护昆达",
    "兰弥那离",
    "兰宁巴",
    "兰沙恭",
    "兰提沙",
    "兰沙陀",
    "兰沙诃",
    "兰耶多",
    "兰阐荼",
    "兰钵答",
    "兰耆都",
    "兰陀尼什",
    "兰帝裟",
    "兰拉吉",
    "兰玛哈",
    "兰雅玛",
    "兰卑浮",
    "兰玛尼",
    "兰羯磨",
    "兰随尼",
    "兰耶师",

    # 4.0 新NPC
    '莉雅丝',
    '布蕾莘',
    '梅雅德',
    '加维',
    "鲁道夫",
    "奥特",
    "卡隆",
    "巴尔塔萨",
    "多洛拉",

    # npc from bwiki spider
    '龙二',
    '齐里亚布',
    '齐米娅',
    '齐米亚',
    '齐亚德',
    '黛安',
    '黑田',
    '黑泽京之介',
    '黄巨贾',
    '麦尔斯',
    '麦克',
    '鹿野奈奈',
    '鹰司朝秀',
    '鹤立',
    '鸿如',
    '鲸井椛',
    '鲁梅拉',
    '鲁克沙',
    '鬼伯伯',
    '高美',
    '高桥',
    '高斐',
    '高哈尔',
    '骆丰',
    '马齐尔',
    '马鲁夫',
    '马茨',
    '马苏迪',
    '马斯鲁尔',
    '马尼特',
    '马塞拉',
    '马塞尔',
    '马克西姆',
    '风速枣椰',
    '须美',
    '顺吉',
    '露泽妮',
    '露子',
    '霖铃',
    '霍夫曼',
    '雷诺德',
    '雷蒙德',
    '雷萨',
    '雪怪伯',
    '雅瓦娜尼',
    '雅丝敏',
    '雄三',
    '陆',
    '阿飞',
    '阿釜',
    '阿金',
    '阿里娜',
    '阿茂',
    '阿芷',
    '阿纳托尔',
    '阿笨',
    '阿直',
    '阿瑟尔',
    '阿玛兹亚',
    '阿泰',
    '阿波',
    '阿法纳西',
    '阿桂',
    '阿望',
    '阿昊',
    '阿旭',
    '阿斯法德',
    '阿敬',
    '阿拉耶什',
    '阿拉比',
    '阿扣',
    '阿托',
    '阿扎莱',
    '阿德菲',
    '阿幸',
    '阿巴尼斯',
    '阿山婆',
    '阿尔明',
    '阿尔弗雷德',
    '阿尔尼姆',
    '阿宽',
    '阿妮萨',
    '阿夫塔',
    '阿大',
    '阿南德',
    '阿凡',
    '阿克拉姆',
    '阿佑',
    '阿伟',
    '阿什莎布',
    '阿什亚',
    '阿亚德瓦',
    '阿二',
    '阿义',
    '阿三',
    '阳介',
    '门达斯',
    '长顺',
    '长门幸子',
    '长野原龙之介', # 宵宫！！！
    '镇海',
    '银杏',
    '铁骨双刀',
    '铁膀子',
    '铁弘',
    '铁也',
    '钱眼儿',
    '鉴秋',
    '金钟',
    '重佐',
    '里瑟',
    '里夫',
    '里卡尔',
    '里凯蒂',
    '迪特玛尔',
    '迪拉瓦尔',
    '辉少',
    '辉子',
    '路爷',
    '赫塔',
    '赛芭',
    '贾米',
    '贾瓦希尔',
    '贾汉吉尔',
    '贾扬特',
    '费雷斯特',
    '费罗兹',
    '费恩',
    '贡托雷',
    '贝鲁兹',
    '贝雅特丽奇',
    '贝格艾蒂',
    '贝拉特',
    '贝尔福',
    '贝尔卡塞姆',
    '贝哈姆',
    '豁牙子',
    '谢赫祖拜尔',
    '诺莱特',
    '诺拉',
    '诚二',
    '诏勤',
    '言笑',
    '观海',
    '西西',
    '西格',
    '西口',
    '街头艺人',
    '虎之助',
    '藤原俊子',
    '薮木',
    '薇塔',
    '蕾欧妮',
    '蕾娅',
    '蔡寻',
    '蔡乐',
    '蒂亚',
    '葫芦',
    '落霞',
    '落',
    '萩原',
    '萨蒂',
    '萨维尼恩',
    '萨纳德',
    '萨拉尔',
    '萨尼娅',
    '萨尔姆',
    '萨宁',
    '萨哈尔',
    '菲尔戈黛特',
    '菲利斯·尤格',
    '菲利克斯',
    '莺儿',
    '莲塔',
    '莱斯',
    '莱弥娅',
    '莱内',
    '莱克图尔',
    '莫约',
    '莫妮欧',
    '莎菲亚',
    '莎菲',
    '莎莎妮',
    '莎莎',
    '莎莉',
    '莎梅耶',
    '莎塔',
    '莉拉',
    '莉兹',
    '荷尔德林',
    '荫山',
    '荒谷',
    '茹斯托',
    '茱塔',
    '茉莉',
    '茅葺一庆',
    '范兵卫',
    '范二爷',
    '茂才公',
    '若紫',
    '苏莱卡',
    '芷若',
    '花初',
    '芭努',
    '芭别尔',
    '芝拉',
    '芙蓉',
    '艾菲',
    '艾莉亚',
    '艾维娜',
    '艾登',
    '艾琪诺',
    '艾珂',
    '艾方索',
    '艾拉姆',
    '艾希拉',
    '艾尔菲',
    '艾尔希娜',
    '艾嘉莉亚',
    '艾伯特',
    '舍万',
    '胡马延',
    '胡尚',
    '耶沙法特',
    '考特里亚',
    '考什克',
    '老高',
    '老赵',
    '老芬奇',
    '老臭',
    '老章',
    '老孙',
    '老图',
    '老周叔',
    '老刘伯',
    '翠儿',
    '羽生田千鹤',
    '罗莎娜',
    '罗斯玛丽',
    '罗小妹',
    '罗因贾',
    '罗伯特',
    '绮珊',
    '绮命',
    '绘星',
    '绀田传助',
    '纳里曼',
    '纳赫蒂加尔',
    '纳杰特',
    '纳吉斯',
    '纱江',
    '纪尧姆',
    '约顿',
    '约里奥',
    '约哈南',
    '红豆',
    '索赫尔',
    '索瑞',
    '索拉雅',
    '索希',
    '篠冢',
    '笹野',
    '笛瑟',
    '竺子',
    '竟达',
    '立本',
    '穆纳切洛',
    '穆甘纳',
    '稻叶久藏',
    '程杆子',
    '秦夫人',
    '科赛尔',
    '秋蔚',
    '秋歌',
    '秀华',
    '福阿德',
    '福迪尔',
    '福图赫',
    '祖尔宛',
    '碧波',
    '石榴',
    '石川',
    '石壮',
    '矢田幸喜',
    '眼尖夜鸦',
    '真田',
    '直江久政',
    '盖伊',
    '皮卡',
    '百合华',
    '白井',
    '留云借风真君',
    '畅畅',
    '男子汉杰克',
    '由宇',
    '田铁嘴',
    '甜甜',
    '甄强',
    '瓦菲格',
    '瓦莱',
    '瓦希德',
    '瓦尔坦',
    '瓦妮塔',
    '璐璐',
    '璃彩',
    '瑾武',
    '琼斯',
    '琪娜特',
    '理水叠山真君',
    '珊瑚',
    '珀姆',
    '玲菜',
    '玫莉莎',
    '玛达赫',
    '玛赫菲',
    '玛蒂哈',
    '玛文',
    '玛希德',
    '玛塔莉',
    '王扳子',
    '王平安',
    '玄冬林檎',
    '玄三',
    '独眼小僧',
    '独孤朔',
    '狗三儿',
    '狐妖',
    '特纳',
    '特洛黛',
    '牙',
    '片山',
    '爱贝尔',
    '爱洛芙',
    '爱洛依丝',
    '爱丝蒂',
    '烦恼夜鸦',
    '潮汐',
    '满',
    '滑头鬼',
    '温克尔',
    '渡边',
    '清涟',
    '清昼',
    '清子',
    '淮安',
    '淑之',
    '海龙',
    '海迪夫',
    '海莉',
    '海老名权四郎',
    '海娜',
    '海妮耶',
    '海伦',
    '海亚姆',
    '浩仔',
    '洛达蒙特',
    '洛薇',
    '泽轩',
    '泽维尔',
    '波尔托',
    '法里芭',
    '法里德',
    '法莎',
    '法莉哈',
    '法斯宾',
    '法拉纳兹',
    '法尔扎妮',
    '法哈德',
    '法加娜',
    '法伽尼',
    '治平',
    '河童',
    '沛休',
    '沙蒂尔',
    '沙瓦玛',
    '沙班达尔',
    '沙欣',
    '沙普尔',
    '沙扬',
    '沙尔玛',
    '沐晨',
    '沐宁',
    '沉香',
    '沃伦·努维尔勋爵',
    '汤雯',
    '汤米',
    '江舟',
    '江城',
    '汝英',
    '汐零',
    '水田',
    '毗伽尔',
    '比塔',
    '毓华',
    '毅',
    '武田',
    '步云',
    '歌德',
    '欧菲妮',
    '欧芙',
    '横山',
    '楠塔克',
    '楚婆婆',
    '楚仪',
    '森口',
    '梶',
    '梨绘',
    '梢',
    '桥西',
    '桑顿',
    '桑格内蒂',
    '桑杰',
    '桑塔利耶',
    '格雷戈尔',
    '格雷戈',
    '格蕾希艾',
    '格莱丝',
    '格奥尔格',
    '柴门惠理',
    '柴门克巳',
    '柴门二郎',
    '柴田',
    '柳德米拉',
    '柯蕾特莉',
    '柯妮莉娅',
    '柯奈尔',
    '查耶维奇',
    '枭总管',
    '枫',
    '松田',
    '松平',
    '松川宗全',
    '松前',
    '杰克',
    '杜兰',
    '村上',
    '李锤子',
    '李晓',
    '李当',
    '李小虎',
    '李九郎',
    '李丁',
    '杉山',
    '朱老板',
    '朝雾',
    '朝仓',
    '月疏',
    '曲清',
    '智树',
    '普里亚',
    '普热瓦',
    '普吕尼埃',
    '晓月左卫门十藏',
    '星辰君',
    '星稀',
    '星火',
    '昌虎',
    '新之丞',
    '斯坦利',
    '斑目百兵卫',
    '效言',
    '撒比特',
    '摩可',
    '摆渡人',
    '振翔',
    '拉赫曼',
    '拉贾维',
    '拉菲克',
    '拉米雅',
    '拉米兹',
    '拉姆齐',
    '扎蒂斯',
    '扎莱',
    '扎卡里亚',
    '扎凯',
    '扎伊',
    '才藏',
    '户田',
    '戴维',
    '戟',
    '戚定',
    '戈尔珊',
    '慧心',
    '慕胜',
    '悬帆',
    '悦子',
    '恩内斯特',
    '思思',
    '思妤',
    '快腿罗斯',
    '忠明',
    '忠常',
    '志村勘兵卫',
    '志华',
    '德长',
    '德赛',
    '德贵',
    '德沃沙克',
    '御舆源次郎',
    '御舆源一郎',
    '徐六石',
    '彦博',
    '强叔',
    '弥诺',
    '弥生七月',
    '弘毅',
    '弗莱琪',
    '弗朗西斯',
    '弗拉德',
    '弗丽德',
    '康纳',
    '应公',
    '库马',
    '库拉什',
    '库尔苏姆',
    '庆兴',
    '广海',
    '平野',
    '平海',
    '平冢',
    '平八',
    '平井',
    '帕维尔',
    '帕纳',
    '帕森',
    '帕斯里',
    '希忒',
    '希尔维娅',
    '希尔米',
    '布鲁斯',
    '布希柯',
    '巴巴尔',
    '巴巴克',
    '巴巴',
    '巴尔马克',
    '巴哈拉克',
    '巴兰',
    '左右加',
    '岩田',
    '岛政兴',
    '山城健太',
    '山上',
    '尼达尔',
    '尼扎姆',
    '尼姆',
    '尼卡',
    '尤骏',
    '尤夫腾',
    '尤南',
    '尤努斯',
    '尤兹贾尼',
    '小雨',
    '小雀儿',
    '小野田',
    '小野',
    '小蒙',
    '小茜',
    '小羽',
    '小绿',
    '小猛',
    '小林',
    '小月',
    '小昭',
    '小春',
    '小星',
    '小征',
    '小山',
    '小安',
    '小姜',
    '小卷',
    '小仓澪',
    '小仓优',
    '小九九',
    '寺田',
    '富贵',
    '寅杰',
    '宫崎三朗',
    '宝儿',
    '宜年',
    '宏达',
    '宏朗',
    '宏宇',
    '宏一',
    '安藤',
    '安蒂拉',
    '安田',
    '安特曼',
    '安洁莉可',
    '安托万',
    '安德烈',
    '安娜',
    '安内特',
    '安东尼',
    '孙宇',
    '婕德',
    '娜蒂亚',
    '娜泽宁',
    '娜比雅',
    '娜扎法琳',
    '娜丝琳',
    '威拉格',
    '妲卡玛忒',
    '妮维妲',
    '奥雷乌斯',
    '奥贝德',
    '奥菲利亚',
    '奥格劳',
    '奥拉夫',
    '奥妮',
    '奥古斯都·洛夫莱斯',
    '奥利弗',
    '奥乌兹',
    '奈拉',
    '奇拉',
    '太郎丸',
    '太田太郎',
    '天野',
    '天奕',
    '大邋遢',
    '大迷糊',
    '大脚',
    '大泉',
    '大河原五右卫门',
    '大武',
    '大岛纯平',
    '大壮',
    '大坤',
    '大和田',
    '大久保大介',
    '多加',
    '夙凌',
    '夏特莱',
    '夏姆',
    '夏妮',
    '墨田',
    '塞德娜',
    '塞库菈',
    '塞塔蕾',
    '塔维妮儿',
    '塔米娜',
    '塔玛拉',
    '塔希尔',
    '塔尔瓦',
    '塔列辛',
    '基翁',
    '埃蒂安',
    '埃泽',
    '埃尔庇艾',
    '埃尔凡',
    '坦吉',
    '图曼',
    '嘉铭',
    '嘉玮',
    '嘉义',
    '唐无仇',
    '唐娜',
    '哲远',
    '哈马维',
    '哈里斯',
    '哈贾娜德',
    '哈立德',
    '哈瓦德',
    '哈瓦',
    '哈琳',
    '哈桑',
    '哈扬',
    '哈巴奇',
    '哈尔瓦尼',
    '哈尔瓦',
    '哈坎',
    '哈勒敦',
    '哈兹姆',
    '和泉那希',
    '和声',
    '吴船长',
    '向明',
    '吉盖克斯',
    '吉丽安娜',
    '叶名山薰',
    '叶卡捷琳娜',
    '古思塔斯普',
    '古尔根',
    '口渴夜鸦',
    '双叶',
    '厄桑',
    '卢克',
    '卡里玛',
    '卡里尔',
    '卡瓦',
    '卡琵莉亚',
    '卡玛尔',
    '卡桑',
    '卡格姆尼',
    '卡斯帕',
    '卡姆拉',
    '卡姆兰',
    '卡塔扬',
    '卡嘉妮',
    '卜劳恩',
    '博行',
    '博彦',
    '博尼法兹',
    '卓英',
    '卉卉',
    '千叶',
    '勒莫迪埃',
    '勒杰',
    '劳伦斯',
    '努查赫',
    '加萨尼',
    '加福尔',
    '加扎里',
    "利涅尔",
    '利弗',
    '切瑟尔',
    '凯撒',
    '凯叔',
    '凯万',
    '冈崎绘里香',
    '内海',
    '内森',
    '内尔敏',
    '其老爷',
    '关宏',
    '关垂',

    # # 角色界面，建议不要用，纯纯污染
    # "元素充能效率",
    # "生命值",
    # "合成次数1",
    # "暴击率",
    # "攻击力",
    # "暴击伤害",
    # "防御力",

    # # 打反应的词
    # "蒸发",
    # "潮湿",
    # "冻结",
]


def random_character_name():
    return random.choice(characters_name)

