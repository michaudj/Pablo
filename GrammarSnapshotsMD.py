# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:00:55 2023

@author: jmd01
"""


from MyLearners import Learner,RWLearner
from Raw_input import Raw_input, ProbabilisticGrammar
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as cf
from datetime import datetime
#import pandas as pd
start_time = datetime.now()
import scipy

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def logistic(x,k,x0):
    return  1/(1+np.exp(-k*(x-x0))) 


def run_simulation(l):
    l.learn(stimuli_stream)
    return l

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
#       Setting learning parameters
#
###############################################################

# Set the initial learning parameters
Learner.initial_value_border = 1.
Learner.initial_value_chunking = -1.
Learner.ID = 0

# Set the parameters controlling reinforcement learning
Learner.alpha = 0.1
Learner.beta = 1.0
Learner.positive_reinforcement = 25.
Learner.negative_reinforcement = -10.

RWLearner.initial_value_border = 1.
RWLearner.initial_value_chunking = -1.

RWLearner.alpha = 0.1
RWLearner.beta = 1.0
RWLearner.positive_reinforcement = 25.
RWLearner.negative_reinforcement = -10.


###############################################################
#
#       Definition of the language using a probabilistic CFG
#
###############################################################

print('Defining the grammar')

# definition of the grammar
# Vocabulary
#############################################################
#
#       Simple grammar with mono and ditransitive verbs
#
#############################################################

number_of_verbs = 1
number_of_nouns = 5
number_of_adj = 1
number_of_relpron = 1
number_of_det = 1
number_of_prep = 1
number_of_monotransitive_verbs = 1
number_of_ditransitive_verbs = 1

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]
prep = ['p' + str(i) for i in range(1, number_of_prep+1)]
monotransitive_verbs = ['mv' + str(i) for i in range(1, number_of_monotransitive_verbs+1)]
ditransitive_verbs = ['dv' + str(i) for i in range(1, number_of_ditransitive_verbs+1)]



terminalsYP = flatten([monotransitive_verbs,ditransitive_verbs,nouns,adjs,relpron,det,prep])
non_terminalsYP = ['S', 'N','NP','VP','V','rel','MV','DV','NPV','AP','PP','R','A','D','P']



production_rulesYP = {
    'S': [['NP', 'VP']],
    'NP': [['N']],#,['D','N'],['D','AP','N'],['N','PP']],
    'VP': [['MV','NP'],['DV','NP','NP']],
    'AP': [['A'],['A','A' ] ],
    'PP': [['P','N']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
    'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
    'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
    'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
    'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)],
    'MV': [['mv' + str(i)] for i in range(1, number_of_monotransitive_verbs+1)],
    'DV': [['dv' + str(i)] for i in range(1, number_of_ditransitive_verbs+1)]
}

weightsYP = {
    'S': [1.0],
    'NP': [1.],
    'VP': [.5,.5],
    'AP': [.5,.5 ],
    'PP': [1.0],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
    'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
    'D': [1/number_of_det for i in range(1, number_of_det+1)],
    'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
    'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    'MV': [1/number_of_monotransitive_verbs for i in range(1, number_of_monotransitive_verbs+1)],
    'DV': [1/number_of_ditransitive_verbs for i in range(1, number_of_ditransitive_verbs+1)]
    }

cfgNVNMD = ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)


#############################################################
#
#       Initializing the learners
#
#############################################################


n_trials = 50010

#############################################################
#
#   Creating the stimuli stream
#
#############################################################
print('Creating the stimuli stream')

# Create stimuli stream
stimuli_stream = Raw_input(10*n_trials,cfgNVNMD)

#############################################################
#
#   Learning snapshots
#
#############################################################

print('Running the model')
snaptid = [500,1000,1500,3000,6000,25000,50000]


print('RW Q-learning, continuous')
learner = RWLearner(n_trials = n_trials, border = 'cont')
learner.learn_with_snapshot(stimuli_stream, 'RWQLearnerC_MD.xlsx', snaptid, 10)


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

