black_list = set([
'(Test)平原料理任务NPC',
'(Test)捕鱼挑战NPC',
'(Test)星火',
'(Test)桥西',
'(Test)漂流瓶收集者',
'(Test)索拉雅',
'(Test)车夫',
'(test）拳手一号',
'(test）盗宝团电',
'(test）镜头',
'???',
'NPC动作测试1-男',
'NPC动作测试2-女',
'NPC动作测试3-女',
'Name',
'PhatomSamurai',
'test伪装的千岩军1',
'test伪装的千岩军2',
'test望舒客栈的搬运工',
'七七线盗宝团1',
'七七线盗宝团2',
'丘丘人1',
'丘丘人2',
'乌鸦2',
'乌鸦3',
'兰那罗A',
'兰那罗B',
'兰那罗C',
'动作测试用派蒙1',
'吟游诗人C',
'和谈现场的反抗军士兵1',
'和谈现场的反抗军士兵2',
'和谈现场的反抗军士兵3',
'和谈现场的反抗军士兵4',
'和谈现场的反抗军士兵5',
'和谈现场的幕府士兵1',
'和谈现场的幕府士兵2',
'和谈现场的幕府士兵3',
'和谈现场的幕府士兵4',
'对战的反抗军1',
'对战的幕府军1',
'幻影路人A',
'幻影路人B',
'幻影路人C',
'幻影路人D',
'愚人众守卫6',
'托马coop围观npc1',
'托马coop围观npc2',
'托马coop围观npc3',
'托马coop围观npc4',
'托马coop围观npc5',
'托马coop学员女1',
'托马coop学员女2',
'托马coop学员女3',
'托马coop学员男1',
'托马coop学员男2',
'托马coop清扫挑战npc1',
'托马coop清扫挑战npc2',
'昆恩（事件测试）',
'气泡B',
'气泡C',
'气泡E',
'璃月嘉宾A',
'璃月嘉宾B',
'社奉行武士B',
'社奉行武士C',
'祭典氛围女A',
'祭典氛围女B',
'祭典氛围女C',
'祭典氛围男A',
'祭典氛围男C',
'蜂巢心声1',
'蜂巢心声10',
'蜂巢心声2',
'蜂巢心声3',
'蜂巢心声4',
'蜂巢心声5',
'蜂巢心声6',
'蜂巢心声7',
'蜂巢心声8',
'蜂巢心声9',
'被骂的反抗军士兵A',
'被骂的反抗军士兵B',
'谜之声A',
'谜之声B',
'谜之声D',
'（事件Test)订餐人',
'西拉杰02',
'丘丘人3',
'「贪婪」的守卫3',
'气泡D',
'乌鸦1',
'气泡A',
'愚人众先遣队员乙',
'祭典氛围男B',
'社奉行武士D',
'愚人众守卫5',
'黄麻子众人2',
'对战的幕府军2',
'对战的反抗军2',
'刁民3',
'谜之声C',
'吟游诗人B',
'社奉行武士A',
'黄麻子众人1',
'镀金旅团C',
'「胆怯」的守卫4',
])

npc_url = "https://raw.githubusercontent.com/DGP-Studio/Snap.Metadata/main/CheatTable/CHS/Npc.csv"
transformer_url = "https://raw.githubusercontent.com/DGP-Studio/Snap.Metadata/main/CheatTable/CHS/Transformer.csv"

# request for the csvs

import requests

npc_csv = requests.get(npc_url)
transformer_csv = requests.get(transformer_url)

npc_csv_lines = npc_csv.text.splitlines()
transformer_csv_lines = transformer_csv.text.splitlines()

npc_names = set()
transformer_names = set()

def valid(s: str) -> bool:
    if '(' not in s and ')' not in s and ' ' not in s and '{' not in s \
        and '?' not in s and s!="PhatomSamurai" and s!="Item" and s not in black_list:
        return True
    else:
        return False

for line in npc_csv_lines:
    npc_names.add(line.split(',')[1])
for line in transformer_csv_lines:
    transformer_names.add(line.split(',')[0])

from mona.text import ALL_NAMES

ALL_NAME = set(ALL_NAMES)
# find out the not-in names

diff_npc = npc_names - ALL_NAME
diff_transformer = transformer_names - ALL_NAME

for dn in diff_npc:
    if not valid(dn):
        continue
    print(f"'{dn}',")
print()
for dt in diff_transformer:
    if not valid(dt):
        continue
    print(f"'{dt}',")