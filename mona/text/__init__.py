from .artifact_name import monster_artifact_name, check_point_artifact_names, treasure_artifact_names
from .characters import characters_name
from .domains import domain_names
from .material import material_names
from .operations import operations_names
from .weapons import weapons_name


lexicon = set({})
for name in monster_artifact_name \
    + check_point_artifact_names \
    + treasure_artifact_names \
    + characters_name \
    + domain_names \
    + material_names \
    + operations_names \
    + weapons_name:
    for char in name:
        lexicon.add(char)


lexicon = sorted(list(lexicon))

index_to_word = {
    0: "-"
}
word_to_index = {
    "-": 0
}
for index, word in enumerate(lexicon):
    index_to_word[index + 1] = word
    word_to_index[word] = index + 1

print(f"lexicon size: {len(word_to_index)}")
