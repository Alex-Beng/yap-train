import random

from .material_names import material_names

def random_material_name():
    return random.choice(material_names)

def random_cant_hold_material():
    return "已不能持有更多的"+ random.choice(material_names) + "，请清理出足够的空间再试"