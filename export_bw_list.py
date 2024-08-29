# from mona.text import bk_list
from mona.text.characters import characters_name
from mona.text.operations import operations_names
from mona.text.domains import domain_names
from mona.text.server_leak_names import server_leak_names
from mona.text.field_operations import field_operations_names

import json
import sys
import os

def read_or_create_json(path: str):
    if os.path.exists(path):
        return json.load(open(path, 'r', encoding='utf-8'))
    else:
        return []

if len(sys.argv) < 2:
    print('Usage: python export_bw_list.py [output_path]')
    exit(1)
pt = sys.argv[1]
path =  os.path.join(pt, 'default_black_list.json')
black_lists = read_or_create_json(path)
# print(black_lists)

# merge the npc and domain and op name to balck lists
all_bk = characters_name + operations_names + domain_names + server_leak_names + field_operations_names
for name in all_bk:
    if name not in black_lists:
        black_lists.append(name)
print(black_lists)
json.dump(black_lists, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

