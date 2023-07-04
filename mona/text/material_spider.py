# crawling Bwiki for material names

import json
import requests
from lxml import etree


def get_html(url):
    try:
        r = requests.get(url, timeout=7)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
    except:
        return ""


if __name__ == "__main__":
    html = get_html("https://wiki.biligame.com/ys/材料图鉴")
    html = etree.HTML(html)
    material_as = html.xpath('//*[@id="mw-content-text"]//div[@class="ys-iconLarge"]/a[1]')
    material_names = [a.get('title') for a in material_as]
    # save to python file
    with open('./material_names.py', 'w', encoding='utf-8') as f:
        f.write(f'material_names = {material_names}')
    # print(len(material_as), type(material_as[0]), type(material_as))

    