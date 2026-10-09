import re

from mona.text import ALL_NAMES
from mona.text.server_leak_names import server_leak_names


npc_names_path = 'npc_names_CHS.txt'
target_path = 'mona/text/server_leak_names.py'

version = '7.1'


def load_npc_names(path):
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.rstrip('\n')
            while name.endswith('$UNRELEASED'):
                name = name[:-len('$UNRELEASED')]
            name = name.strip()
            if name:
                names.append(name)
    return sorted(set(names))


def valid_word(wd):
    if not wd:
        return False
    if 'test' in wd.lower():
        return False
    if wd.startswith('beyond') or wd.startswith('SneakLE'):
        return False
    if re.match(r'^\d+\.\d', wd):
        return False
    if wd.startswith('#{'):
        return False
    if re.search(r'[A-Za-z]', wd):
        return False
    if re.search(r'[0-9]', wd):
        return False
    if '?' in wd:
        return False
    if any(ch in wd for ch in '()（）{}… \u3000'):
        return False
    return True


def write_names(path, existing, new_names, region_prefix):
    if not new_names:
        print(f'no new names for {region_prefix}')
        return 0

    with open(path, 'r', encoding='utf-8', newline='') as f:
        content = f.read()

    nl = '\r\n' if '\r\n' in content else '\n'
    lines = [f"    '{name}'," for name in new_names]
    block = (
        nl + f'    # region {region_prefix} v{version}（自动生成）'
        + nl + nl.join(lines)
        + nl + f'    # endregion {region_prefix} v{version}' + nl
    )

    # insert before the final closing bracket of the list
    marker = nl + ']'
    idx = content.rindex(marker)
    content = content[:idx] + block + content[idx:]

    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)

    print(f'{region_prefix}: appended {len(new_names)} names to {path}')
    return len(new_names)


if __name__ == '__main__':
    all_names = set(ALL_NAMES)
    existing = set(server_leak_names)
    npc_names = load_npc_names(npc_names_path)
    new_names = [
        it for it in npc_names
        if it not in all_names and it not in existing and valid_word(it)
    ]

    write_names(target_path, existing, new_names, 'npc_names_CHS.txt')
