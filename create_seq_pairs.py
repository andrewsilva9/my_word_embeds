import os
import random
import cPickle
import string
import re
# TODO: currently not preserving conversation context, only message / response. context would help


def clean_message(message):
    # Remove new lines within message
    cleaned = message.replace('\n',' ').lower()
    # Deal with some weird tokens
    cleaned = cleaned.replace("\xc2\xa0", "")
    # Remove punctuation
    cleaned = re.sub('([.,!?])','', cleaned)
    # Remove multiple spaces in message
    cleaned = re.sub(' +',' ', cleaned)
    return cleaned

master_dir = '/Users/andrewsilva/Desktop/master/'
pairs = []
longest = 0
for filename in os.listdir(master_dir):
    master_txt = open(os.path.join(master_dir, filename), 'r').read().lower().split('\n')
    # Remove blank sequences
    seqs_to_remove = []
    for seq_ind in range(len(master_txt)):
        sequence = master_txt[seq_ind]
        sequence_intro = sequence.split(':')[0]
        if sequence_intro.split(' ')[0] == '@':
            continue
        elif len(sequence.split(':')) < 2 or len(sequence.split(':')[1].split(' ')) < 2:
            seqs_to_remove.append(sequence)
            for upcoming_seq in range(seq_ind + 1, len(master_txt)):
                seqs_to_remove.append(master_txt[upcoming_seq])
                if len(master_txt[upcoming_seq].split(' ')) > 2 and master_txt[upcoming_seq].split(' ')[1] == '@':
                    break

    for seq in seqs_to_remove:
        if seq in master_txt:
            master_txt.remove(seq)

    if len(master_txt) < 1:
        continue
    while master_txt[0].split(' ')[0] == 'me:' or master_txt[0].split(' ')[1] == '@':
        master_txt.remove(master_txt[0])
        if len(master_txt) < 1:
            break
    if len(master_txt) < 1:
        continue

    last_speaker = None
    friend_spoken = False
    me_spoken = False
    current_friend = []
    current_me = []
    index = 0

    while index < len(master_txt) - 1:
        sequence = clean_message(master_txt[index])
        split_seq = sequence.split(' ')
        if len(split_seq) < 1:
            print sequence
            index += 1
            continue
        speaker = sequence.split(':')[0]
        if len(speaker.split(' ')) > 1 and speaker.split(' ')[1] == '@':
            index += 1
            continue
        sequence = " ".join(str(x) for x in sequence.split(':')[1:]).split(' ')

        # Friend still speaker:
        if speaker != 'me' and speaker == last_speaker:
            current_friend.extend(sequence)
            friend_spoken = True
        # I'm still speaking:
        elif speaker == 'me' and speaker == last_speaker:
            current_me.extend(sequence)
            me_spoken = True
        # Friend speaking for the first time:
        elif speaker != 'me' and friend_spoken is False:
            current_friend.extend(sequence)
            friend_spoken = True
        # I'm speaking for the first time:
        elif speaker == 'me' and me_spoken is False:
            current_me.extend(sequence)
            me_spoken = True
        # We've both spoken and are on me/friend for the second time
        else:
            # Create pair of friend / my response
            if len(current_me) > longest:
                longest = len(current_me)
            if len(current_friend) > longest:
                longest = len(current_friend)
            input_text = " ".join(str(x) for x in current_friend)
            output_text = " ".join(str(x) for x in current_me)
            pairs.append((input_text, output_text))
            # print pairs
            # last speaker will be friend but whatever, set to none
            # debug_pairs.append((current_friend, current_me))
            current_friend = []
            current_me = []
            last_speaker = None
            friend_spoken = False
            me_spoken = False
            # Redo this sequence because it didn't get added to anything
            index -= 1
        index += 1
        last_speaker = speaker
print len(pairs)
print random.choice(pairs)
print longest
cPickle.dump(pairs, open('pairs.pkl', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
