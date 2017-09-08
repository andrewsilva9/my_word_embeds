import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pickle


def load_checkpoint(filename):
    chk = torch.load(filename)
    return chk


use_gpu = torch.cuda.is_available()

word_to_ix = pickle.load(open('word_to_ix.pkl', 'rb'))


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size, 256)
        self.linear2 = nn.Linear(256, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1, -1)
        return embeds


# here context_size should be 4 instead of 2
embed_dim = 128
cont_size = 4
if use_gpu:
    model = CBOW(len(word_to_ix), embedding_dim=embed_dim, context_size=cont_size).cuda()
    loss_function = nn.NLLLoss().cuda()
else:
    model = CBOW(len(word_to_ix), embedding_dim=embed_dim, context_size=cont_size)
    loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=.001)

chk = load_checkpoint('my_cbow_embedding_model52.pth.tar')
model.load_state_dict(chk['state_dict'])
word_to_embeds = {}
embeds_to_words = {}
for word in word_to_ix.keys():
    model.zero_grad()
    if use_gpu:
        context_vars = autograd.Variable(torch.cuda.LongTensor([word_to_ix[word]]))
    else:
        context_vars = autograd.Variable(torch.LongTensor([word_to_ix[word]]))
    print 'Word:', word
    print 'IX:', word_to_ix[word]
    print 'ContextVars:', context_vars.data[0]
    # FIXME: Space hack. Remove this when I fix word embeddings to work with 0... /sigh
    if word_to_ix[word] == 0:
        continue
    embeds = model(context_vars)
    word_to_embeds[word] = embeds
    embeds_to_words[embeds] = word

pickle.dump(word_to_embeds, open('word_to_embeds.pkl', 'wb'))
pickle.dump(embeds_to_words, open('embeds_to_word.pkl', 'wb'))

    # print 'Embedding:', embeds
