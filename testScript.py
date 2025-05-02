# -*- coding: utf-8 -*-
"""
Created on Thu May  1 22:24:02 2025

@author: jmd01
"""

from RawInput import RawInput, RawInputLazy, ProbabilisticGrammar
from Learner import LearnerConfig, Learner

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

print('Defining the grammar')

# definition of the grammar
# Vocabulary
number_of_verbs = 5
number_of_nouns = 5
number_of_adj = 0
number_of_relpron = 2
number_of_det = 0

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]

terminals2 = flatten([verbs,nouns,adjs,relpron,det])
non_terminals2 = ['S', 'N','NP','VP','V','RelCl']

###############################################################
#
#       NVN language
#
###############################################################

# Grammatical rules
production_rulesNVN = {
    'S': [['N', 'VP','N']],
    'VP': [['V']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsNVN = {
    'S': [1],
    'VP': [1],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

cfgNVN = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVN)
# Context free grammar

stimuli_stream = RawInputLazy(n_sentences=2000, grammar=cfgNVN)

# Configure and initialize learner
config = LearnerConfig(n_trials=1000,
                       border = 'cont',
                       initial_value_chunking=-1.,
                       initial_value_border=1.,
                       alpha= 0.1,
                       beta= 1.9,
                       positive_reinforcement = 25,
                       negative_reinforcement = -10,
                       RW=False)
learner = Learner(config)

# Run learning
learner.learn(stimuli_stream)

def moving_average(data, window_size):
    if len(data) < window_size:
        return []
    return [sum(data[i:i+window_size]) / window_size for i in range(len(data) - window_size + 1)]


import matplotlib.pyplot as plt

# Assume learner.history.success is your list of 0s and 1s
successes = learner.history.success
window_size = 15
ma = moving_average(successes, window_size)

plt.figure(figsize=(10, 5))
plt.plot(ma, label=f'{window_size}-trial Moving Average')
plt.xlabel('Trial')
plt.ylabel('Success Rate')
plt.title('Learning Curve')
plt.grid(True)
plt.legend()
plt.show()


# Print results
#print(f"Learner completed after index: {learner.final_index}")
#print(f"Success history: {learner.history.success}")
#print(f"Sentence lengths: {learner.history.sent_len}")