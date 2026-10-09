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


type_code_suffix = (
    r'女|男|成女|成男|少女|少年|小孩|老人|兽怪|剧团成男|贵族壮汉|贵族成男|'
    r'矮灵女|矮灵男|矮灵|壮汉|风仙女|风仙男|风仙|雪精|胖兽怪|红衣|工程服|仆从'
)


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
    # 过滤遮挡/占位词
    if '■' in wd or '【' in wd or '】' in wd:
        return False
    if '测试' in wd or '废弃' in wd:
        return False
    # 过滤带类型代号后缀的内部角色名，如 剧院观众-成男 / 愚人众-女-红衣
    if re.search(r'-(' + type_code_suffix + r')(-|$)', wd):
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
