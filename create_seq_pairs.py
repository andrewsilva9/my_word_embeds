import os
import random
import pickle
import string
# TODO: currently not preserving conversation context, only message / response. context would help

master_dir = os.path.join(os.getcwd(), 'masters')
pairs = []
longest = 0
for filename in os.listdir(master_dir):
    master_txt = open(os.path.join(master_dir, filename), 'r').read().lower().split('\n')

    # TODO: DEBUG
    debug_pairs = []
    # Remove blank sequences
    for sequence in master_txt:
        if (sequence.split(' ')[0] == 'friend:' or sequence.split(' ')[0] == 'me:') and (len(sequence.split(' ')) <= 2):
            if sequence.split(' ')[1] == '':
                master_txt.remove(sequence)

    # Skip to when friend speaks first so I can model responses
    while master_txt[0].split(' ')[0] != 'friend:':
        master_txt.remove(master_txt[0])
        if len(master_txt) < 1:
            break

    last_speaker = None
    friend_spoken = False
    me_spoken = False
    current_friend = []
    current_me = []
    index = 0

    while index < len(master_txt) - 1:
        sequence = master_txt[index].split().translate
        speaker = sequence[0]

        # If it's a continuation of a person
        if speaker != 'friend:' and speaker != 'me:':
            if last_speaker is None:
                # Should never happen
                index += 1
                continue
            elif last_speaker == 'friend:':
                current_friend.extend(sequence)
                friend_spoken = True
            elif last_speaker == 'me:':
                current_me.extend(sequence)
                me_spoken = True
            index += 1
            continue

        # Remove speaker tag from sequence
        sequence = sequence[1:]

        # Remove punctuation from sequence
        for index in range(len(sequence)):
            sequence[index] = sequence[index].translate(None, string.punctuation)
            # Remove empty spaces where punctuation used to be
            if sequence[index] == '':
                sequence.remove('')
                index -= 1

        if last_speaker is None:
            last_speaker = speaker
        # Friend still speaker:
        if speaker == 'friend:' and speaker == last_speaker:
            current_friend.extend(sequence)
            friend_spoken = True
        # I'm still speaking:
        elif speaker == 'me:' and speaker == last_speaker:
            current_me.extend(sequence)
            me_spoken = True
        # Friend speaking for the first time:
        elif speaker == 'friend:' and friend_spoken is False:
            current_friend.extend(sequence)
            friend_spoken = True
        # I'm speaking for the first time:
        elif speaker == 'me:' and me_spoken is False:
            current_me.extend(sequence)
            me_spoken = True
        # We've both spoken and are on me/friend for the second time
        else:
            # Create pair of friend / my response
            if len(current_me) > longest:
                longest = len(current_me)
            if len(current_friend) > longest:
                longest = len(current_friend)
            pairs.append((current_friend, current_me))
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
pickle.dump(pairs, open('pairs.pkl', 'wb'))
