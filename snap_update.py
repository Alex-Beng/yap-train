

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
    print(f"'{dn}',")
print()
for dt in diff_transformer:
    print(f"'{dt}',")