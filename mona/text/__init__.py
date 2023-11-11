from .artifact_name import monster_artifact_name, check_point_artifact_names, treasure_artifact_names
from .characters import characters_name
from .domains import domain_names
from .material import material_names
from .operations import operations_names
from .weapons import weapons_name
from .server_leak_names import server_leak_names
from .book_names import book_names
from .common_Chinese import common_Chinese

lexicon = set({})
ALL_NAMES = monster_artifact_name \
    + check_point_artifact_names \
    + treasure_artifact_names \
    + characters_name \
    + domain_names \
    + material_names \
    + operations_names \
    + weapons_name \
    + server_leak_names \
    + book_names 

ALL_NAMES_WITH_COMMON_CHINESE = ALL_NAMES.copy()
ALL_NAMES_WITH_COMMON_CHINESE = ALL_NAMES_WITH_COMMON_CHINESE \
    + common_Chinese
namelen2num = {}
for name in ALL_NAMES:
    namelen2num[len(name)] = namelen2num.get(len(name), 0) + 1
    for char in name:
        lexicon.add(char)
lens = sorted(list(namelen2num.keys()))
print(f"len 2 num: {lens}\n{[namelen2num[l] for l in lens]}")

lexicon = sorted(list(lexicon))

index_to_word = {
    0: "|"
}
word_to_index = {
    "|": 0
}
for index, word in enumerate(lexicon):
    index_to_word[index + 1] = word
    word_to_index[word] = index + 1

print(f"lexicon size: {len(word_to_index)}")
