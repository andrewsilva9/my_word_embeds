import numpy as np
import pylab as Plot
import pickle
import tsne as ts


word_to_embed = pickle.load(open('word_to_embeds.pkl', 'rb'))
embed_to_word = pickle.load(open('embeds_to_word.pkl', 'rb'))
target_words = word_to_embed.keys()[:2000]

rows = [word_to_embed[word] for word in target_words if word in word_to_embed]

target_matrix = np.array(rows)
reduced_matrix = ts.tsne(target_matrix, 2)

Plot.figure(figsize=(200, 200), dpi=100)

max_x = Math.amax(reduced_matrix, axis=0)[0]
max_y = Math.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x, max_x))
Plot.ylim((-max_y, max_y))

Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

for row_id in range(0, len(rows)):
    target_word = embed_to_word[rows[row_id]]
    x = reduced_matrix[row_id, 0]
    y = reduced_matrix[row_id, 1]
    Plot.annotate(target_word, (x, y))

Plot.savefig("my_first_2000.png")
