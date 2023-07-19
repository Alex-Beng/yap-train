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
}
word_to_index = {
}
for index, word in enumerate(lexicon):
    index_to_word[index + 1] = word
    word_to_index[word] = index + 1

print(f"lexicon size: {len(word_to_index)}")
