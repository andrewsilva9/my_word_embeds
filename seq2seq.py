import os
import string
import re
import random
import time
import math
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 2000
EMBED_DIM = 128


word_to_embed = pickle.load(open('word_to_embeds.pkl', 'rb'))
embed_to_word = pickle.load(open('embeds_to_word.pkl', 'rb'))
missing_word = word_to_embed['unknown']


def embed_words(sentence_list):
    sequence = []
    for word in sentence_list:
        sequence.append(embed_word(word))
    return sequence


def embed_word(word):
    return word_to_embed.get(word, missing_word)

data_in = pickle.load(open('pairs.pkl', 'rb'))

#
# def variable_from_sentence(sentence):
#     embeds_arr = embed_words(sentence)
#     var = Variable(torch.LongTensor(embeds_arr).view(-1, 1))
#     if use_gpu:
#         var = var.cuda()
#     return var


def variables_from_pair(pair):
    input_var = embed_words(pair[0])
    target_var = embed_words(pair[1])
    return input_var, target_var


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, embed_in, hidden):
        # Note: we run this all at once (over the whole input sequence)
        # embedded = self.embedding(embed_in).view(1, -1)
        output, hidden = self.gru(embed_in, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if use_gpu:
            hidden = hidden.cuda()
        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if use_gpu:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = GeneralAttn(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, embed_in, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(embed_in).view(1, -1)
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


encoder_test = EncoderRNN(EMBED_DIM, 256, 2)
decoder_test = AttnDecoderRNN('general', 256, 256, 2)
print(encoder_test)
print(decoder_test)

encoder_hidden = encoder_test.init_hidden()
print data_in[0]
word_in, target = variables_from_pair(data_in[0])
if use_gpu:
    encoder_test.cuda()
    word_input = [x.cuda() for x in word_in]
for embed in word_in:
    encoder_outputs, encoder_hidden = encoder_test(embed, encoder_hidden)
# encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

decoder_attns = torch.zeros(1, len(word_in), EMBED_DIM)
decoder_hidden = encoder_hidden
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

if use_gpu:
    decoder_test.cuda()
    word_inputs = [x.cuda() for x in word_in]
    decoder_context = decoder_context.cuda()

for index, embed in enumerate(word_in):
    decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(embed, decoder_context, decoder_hidden, encoder_outputs)
    print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
    decoder_attns[0, index] = decoder_attn.squeeze(0).cpu().data
