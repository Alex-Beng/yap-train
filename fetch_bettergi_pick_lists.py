import re

import requests

from mona.text import ALL_NAMES


white_url = 'https://raw.githubusercontent.com/babalae/bettergi-libraries/main/BetterGI.Assets.Other/Assets/Config/Pick/default_pick_white_lists.json'
black_url = 'https://raw.githubusercontent.com/babalae/bettergi-libraries/main/BetterGI.Assets.Other/Assets/Config/Pick/default_pick_black_lists.json'


# 类型代号后缀：内部角色名标记，如 剧院观众-成男 / 愚人众-女-红衣
type_code_suffix = (
    r'女|男|成女|成男|少女|少年|小孩|老人|兽怪|剧团成男|贵族壮汉|贵族成男|'
    r'矮灵女|矮灵男|矮灵|壮汉|风仙女|风仙男|风仙|雪精|胖兽怪|红衣|工程服|仆从'
)


def is_suspicious(wd):
    if not wd:
        return True
    if '■' in wd or '【' in wd or '】' in wd:
        return True
    if '测试' in wd or '废弃' in wd:
        return True
    if re.search(r'-(' + type_code_suffix + r')(-|$)', wd):
        return True
    return False


def fetch_list(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def print_missing(title, names, existing):
    missing = [n for n in names if n not in existing]
    normal = [n for n in missing if not is_suspicious(n)]
    ignored = [n for n in missing if is_suspicious(n)]

    print(f'===== {title} =====')
    print(f'名单共 {len(names)} 条，本工程缺失 {len(missing)} 条 '
          f'(待添加 {len(normal)}，已忽略 {len(ignored)})')
    print()
    print(f'----- {title} 缺失清单（{len(normal)} 条）-----')
    for n in normal:
        print(f'"{n}",')
    print()
    print(f'----- {title} 已忽略（疑似脏词，{len(ignored)} 条）-----')
    for n in ignored:
        print(f'"{n}",')
    print()


if __name__ == '__main__':
    existing = set(ALL_NAMES)

    white = fetch_list(white_url)
    black = fetch_list(black_url)

    print_missing('白名单 default_pick_white_lists.json', white, existing)
    print_missing('黑名单 default_pick_black_lists.json', black, existing)
