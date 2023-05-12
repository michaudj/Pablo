# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:51:55 2019

@author: jermi792
"""

from learner import Learner, Chunk
from new_raw_input import Raw_input
import numpy as np
import matplotlib.pyplot as plt

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

###############################################################
#
#       Definition of the language using a probabilistic CFG
#
###############################################################
print('Defining the grammar')

# definition of the grammar
# Vocabulary
number_of_verbs = 2
number_of_nouns = 2
number_of_adj = 0
number_of_relpron = 1
number_of_det = 0

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]

terminals2 = flatten([verbs,nouns,adjs,relpron,det])
non_terminals2 = ['S', 'N','NP','VP','V']

# Grammatical rules
production_rules2 = {
    'S': [['N', 'VP'],['N','VP','RelCl']],
    'VP': [['V','N'],['V']],
    'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weights = {
    'S': [0.8, 0.2],
    'VP': [0.8, 0.2],
    'RelCl': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

# Grammatical rules
production_rules = {
    'S': [['N', 'VP']],
    'VP': [['V','N']],
    'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}
# Context free grammar
cfg = {
    'terminals': terminals2,
    'non_terminals': non_terminals2,
    'production_rules': production_rules2,
    'weights': weights
}

# Create stimuli stream
stimuli_stream = Raw_input(20000,cfg,'S')

#############################################################
#
#   Specifying properties of learners and initialization of chunks
#
#############################################################
print('Initializing learners properties')

# Select type of chunking mechanism
#typ = 'right'
typ = 'flexible'

# Set the initial learning parameters
Chunk.initial_value_border = 1
Chunk.initial_value_chunking = -1

# Set the parameters controlling reinforcement learning
Learner.alpha = 0.2
Learner.beta = 1
Learner.positive_reinforcement = 5
Learner.negative_reinforcement = -1

#############################################################
#
#       Running the simulation
#
#############################################################
print('Simulations')
# number of simulations
n_sim = 10

# Create as many learners as number of simulations.
learners = [Learner(n_trials = 10000, t=typ) for i in range(n_sim)]

# Run the simulation
count = 0
for l in learners:
    count += 1
    print(count)
    l.learn(stimuli_stream)
    
#############################################################
#
#       Postprocessing
#
#############################################################

# Process the result    
success = 0.*np.array(learners[0].success)
for l in learners:
    success += np.array(l.success)
    
success /= n_sim

# Plot the results
plt.plot(success)