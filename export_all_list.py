from mona.text import ALL_NAMES
import json
import sys
import os

if len(sys.argv) < 2:
    print('Usage: python export_all_list.py [output_path]')
    exit(1)
pt = sys.argv[1]
path =  os.path.join(pt, 'all_list.json')
json.dump(ALL_NAMES, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
