import os
import string
import re
import random
import time
import math
import pickle
import torch
import torch.nn as nn
import torch.autograd as Variable
from torch import optim
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1

# TODO: improve data processing: Fix pairing so that friend / me pairs are actually real (not cross conversation or cross-time)
# Probably need to edit bash script for that? or put all conversations into one big directory and iterate over each conversation


class language:
    def __init__(self, name):
        self.name = name
        self.word_to_embed = pickle.load(open('word_to_embeds.pkl', 'rb'))
        self.embed_to_word = pickle.load(open('embeds_to_word.pkl', 'rb'))
        self.missing_word = self.word_to_embed['unknown']

    def embed_words(self, sentence):
        sequence = [SOS_token]
        for word in sentence.split(' '):
            sequence.append(self.embed_word(word))
        sequence.append(EOS_token)
        return sentence

    def embed_word(self, word):
        return self.word_to_embed.get(word, self.missing_word)

master_dir = os.path.join(os.getcwd(), 'masters')
for filename in os.listdir(master_dir):
    master_txt = open(os.path.join(master_dir, filename), 'r').read().lower().split('\n')
    # Skip to when my friend speaks first so I can model my responses
    for sequence in master_txt:
        if (sequence.split(' ')[0] == 'friend:' or sequence.split(' ')[0] == 'me:') and (len(sequence.split(' ')) <= 2):
            if sequence.split(' ')[1] == '':
                master_txt.remove(sequence)
                print 'seq to remove', sequence
    while master_txt[0].split(' ')[0] != 'friend:':
        master_txt.remove(master_txt[0])
        if len(master_txt) < 1:
            break
    print master_txt[0]
    last_speaker = None
    friend_spoken = False
    me_spoken = False
    pairs = []
    current_friend = []
    current_me = []
    index = 0
    while index < len(master_txt) - 1:
        sequence = master_txt[index].split(' ')
        print 'SEQ', sequence
        speaker = sequence[0]
        print 'Speaker: ', speaker
        print 'Last speaker:', last_speaker
        print 'Sequence:', sequence
        # If it's a continuation of a person
        if speaker != 'friend:' and speaker != 'me:':
            print 'Continuation of:', last_speaker
            if last_speaker is None:
                # Should never happen
                index += 1
                continue
            elif last_speaker == 'friend:':
                current_friend.append(sequence)
                friend_spoken = True
            elif last_speaker == 'me:':
                current_me.append(sequence)
                me_spoken = True
            index += 1
            continue

        # Remove speaker tag from sequence
        sequence = sequence[1:]
        if last_speaker is None:
            last_speaker = speaker
        # Friend still speaker:
        if speaker == 'friend:' and speaker == last_speaker:
            current_friend.append(sequence)
            friend_spoken = True
        # I'm still speaking:
        elif speaker == 'me:' and speaker == last_speaker:
            current_me.append(sequence)
            me_spoken = True
        # Friend speaking for the first time:
        elif speaker == 'friend:' and friend_spoken is False:
            current_friend.append(sequence)
            friend_spoken = True
        # I'm speaking for the first time:
        elif speaker == 'me:' and me_spoken is False:
            current_me.append(sequence)
            me_spoken = True
        # We've both spoken and are on me/friend for the second time
        else:
            # Create pair of friend / my response
            pairs.append((current_friend, current_me))
            # last speaker will be friend but whatever, set to none
            last_speaker = None
            friend_spoken = False
            me_spoken = False
            # Redo this sequence because it didn't get added to anything
            index -= 1
            break
        index += 1
        last_speaker = speaker
        print 'Current Friend:', current_friend
        print 'Current Me:', current_me
    break
print pairs[0]
pickle.dump(pairs, open('pairs.pkl', 'wb'))
