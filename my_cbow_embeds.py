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

#
# def get_rid_of(list_in, word):
#     for item in range(list_in.count(word)):
#            list_in.remove(word)
#     return list_in
# master_txt_file = open('master.txt', 'r')
# print 'file in...'
#
# raw_text = str(master_txt_file.read()).split()
# print 'text split up...'
# print raw_text.count('Me:')
# for i in range(raw_text.count('Me:')):
#     raw_text.remove('Me:')
# print 'Me: removed...'
#
# for i in range(raw_text.count('Friend:')):
#     raw_text.remove('Friend:')
# print 'Friend: removed...'
#
# pickle.dump(raw_text, open('raw_text.pickle', 'wb'))

raw_text = pickle.load(open('raw_text.pickle', 'rb'))
word_to_ix = {}
for i, word in enumerate(raw_text):
    word = word.lower()
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix) + 1
print 'set created...'
data = []
# the original implementation will introduce some out-of-range index
# so let's make it consective to avoid such issue
for i in range(2, len(raw_text)-2):
    context = [raw_text[i-2].lower(), raw_text[i-1].lower(),
                raw_text[i+1].lower(), raw_text[i+2].lower()]
    target = raw_text[i].lower()
    data.append((context, target))

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
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

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

# chk = load_checkpoint('my_cbow_embedding_model100.pth.tar')
# curr_epoch = chk['epoch']
# model.load_state_dict(chk['state_dict'])
# optimizer.load_state_dict(chk['optimizer'])
curr_epoch = 0
while curr_epoch < 100:
    total_loss = torch.cuda.FloatTensor([0])
    iter = 0
    for context, target in data:
        iter += 1
        model.zero_grad()

        if use_gpu:
            context_vars = autograd.Variable(torch.cuda.LongTensor([word_to_ix[word] for word in context]))
        else:
            context_vars = autograd.Variable(torch.LongTensor([word_to_ix[word] for word in context]))

        log_probs = model(context_vars)

        if use_gpu:
            loss = loss_function(log_probs, autograd.Variable(torch.cuda.LongTensor([word_to_ix[target]])))
        else:
            loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if iter % print_every == 0:
            print "iteration:", iter, "loss:", loss.data[0]
    losses.append(total_loss)
    curr_epoch += 1
    save_checkpoint({
        'epoch': curr_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename='my_cbow_embedding_model'+str(curr_epoch)+'.pth.tar')
print losses
