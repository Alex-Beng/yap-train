import requests
from lxml import etree

from mona.text.material import material_names


material_url = 'https://wiki.biligame.com/ys/%E6%9D%90%E6%96%99%E5%9B%BE%E9%89%B4'
target_path = 'mona/text/material.py'

version = '7.1'

ignore = {
    '首页',
    '创建新页面',
    '材料一览',
}


def fetch_titles(url):
    r = requests.get(url, timeout=15)
    r.encoding = r.apparent_encoding
    html = etree.HTML(r.text)
    titles = html.xpath('//*[@id="mw-content-text"]//a/@title')
    unique = []
    seen = set()
    for t in titles:
        t = t.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        unique.append(t)
    return unique


def write_names(path, new_names, region_prefix):
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

    marker = nl + ']'
    idx = content.rindex(marker)
    content = content[:idx] + block + content[idx:]

    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)

    print(f'{region_prefix}: appended {len(new_names)} names to {path}')
    return len(new_names)


if __name__ == '__main__':
    existing = set(material_names)
    titles = fetch_titles(material_url)
    new_names = [t for t in titles if t not in ignore and t not in existing]

    write_names(target_path, new_names, 'bwiki 材料图鉴')
