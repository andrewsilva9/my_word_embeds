import pickle


def get_rid_of(list_in, word):
    for item in range(list_in.count(word)):
           list_in.remove(word)
    return list_in

master_txt_file = open('master.txt', 'r')
print 'file in...'
raw_text = str(master_txt_file.read()).split()

for index in range(len(raw_text)):
    raw_text[index] = raw_text[index].lower()

pickle.dump(raw_text, open('raw_text.pkl', 'wb'))

word_to_ix = {' ': 0}
for i, word in enumerate(raw_text):
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)

pickle.dump(word_to_ix, open('word_to_ix.pkl', 'wb'))
