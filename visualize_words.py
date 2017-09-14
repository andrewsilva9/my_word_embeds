import numpy as np
import matplotlib.pyplot as plt
import pickle
import tsne as ts


word_to_embed = pickle.load(open('cpustuff/cpuwords.pkl', 'rb'))
embed_to_word = pickle.load(open('cpustuff/wordscpu.pkl', 'rb'))
for key in word_to_embed.keys():
    if 'http' in key or '\\x' in key:
        del word_to_embed[key]

target_words = word_to_embed.keys()[:2000]

rows = [word_to_embed[word].cpu().data.numpy()[0] for word in target_words if word in word_to_embed]

target_matrix = np.array(rows)
reduced_matrix = ts.tsne(target_matrix, 2)

plt.figure(figsize=(200, 200), dpi=100)

max_x = np.amax(reduced_matrix, axis=0)[0]
max_y = np.amax(reduced_matrix, axis=0)[1]
plt.xlim((-max_x, max_x))
plt.ylim((-max_y, max_y))

plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20)

for row_id in range(0, len(rows)):
    target_word = target_words[row_id]
    x = reduced_matrix[row_id, 0]
    y = reduced_matrix[row_id, 1]
    try:
        plt.annotate(target_word, (x, y))
    except:
        print 'emoji maybe'
        continue

plt.show()
plt.savefig("my_first_2000.png")
