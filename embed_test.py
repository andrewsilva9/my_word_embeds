import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

def save_checkpoint(state, filename='my_cbow_embedding_model.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(filename):
    chk = torch.load(filename)
    return chk


use_gpu = torch.cuda.is_available()

raw_text = pickle.load(open('raw_text.pickle', 'rb'))
word_to_ix = {}
for i, word in enumerate(raw_text):
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix) + 1
print 'set created...'
# print data[:5]
print_every = 1000


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size, 256)
        self.linear2 = nn.Linear(256, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1, -1)
        return embeds


losses = []
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

chk = load_checkpoint('my_cbow_embedding_model100.pth.tar')
curr_epoch = chk['epoch']
model.load_state_dict(chk['state_dict'])
optimizer.load_state_dict(chk['optimizer'])

for word in word_to_ix.keys():
    model.zero_grad()
    # model.test()
    if use_gpu:
        context_vars = autograd.Variable(torch.cuda.LongTensor(word_to_ix[word]))
    else:
        context_vars = autograd.Variable(torch.LongTensor(ix))
    print 'Word:', word
    print 'IX:', word_to_ix[word]
    if word_to_ix[word] == 0:
        continue
    embeds = model(context_vars)
    # print 'Embedding:', embeds
