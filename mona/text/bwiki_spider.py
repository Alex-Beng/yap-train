# 已弃用
'''
crawling Bwiki for names
1. materials
2. artifacts
3. NPCs
4. operations
5. charaters

only 1&2 needed
actually only 1 needed
'''

import requests
from lxml import etree


def get_html(url):
    try:
        r = requests.get(url, timeout=7)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
    except:
        print("wrong")
        return ""


if __name__ == "__main__":
    html = get_html("https://wiki.biligame.com/ys/兰那罗")
    html = etree.HTML(html)
    material_as = html.xpath('//*[@id="mw-content-text"]//td/a[1]/text()')
    # //*[@id="CardSelectTr"]/tbody/tr[1]/td[1]/a
    # 
    # 材料图鉴              //*[@id="mw-content-text"]//div[@class="ys-iconLarge"]/a[1]
    # npc, 除了第一个龙二   //*[@id="frameNpc"]//div[@class="giconCard"]/a[1] 
    print(len(material_as))
    material_names = [a for a in material_as]
    # save to python file
    with open('./lan_names.py', 'w', encoding='utf-8') as f:
        f.write(f'material_names = {material_names}')
    # print(len(material_as), type(material_as[0]), type(material_as))

    