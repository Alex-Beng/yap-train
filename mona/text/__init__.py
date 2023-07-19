from .artifact_name import artifact_name
from .stat import stat_name
from .characters import characters_name
from .material import material_names


lexicon = set({})
for name in material_names:
    for char in name:
        lexicon.add(char)
for name in artifact_name:
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

# 527 -> 980
# 试试能不能收敛
# train only material: 668
# material + artifact: 876
print(f"lexicon size: {len(word_to_index)}")
