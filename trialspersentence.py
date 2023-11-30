# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:29:27 2023

@author: jmd01
"""

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
# do your work here

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def logistic(x,L,k,x0):
    return  L/(1+np.exp(-k*(x-x0))) 


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


    
def get_success_and_length(learners):
    success = 0.*np.array(learners[0].success)
    sent_len = 0.*np.array(learners[0].sent_len)
    for l in learners:
        success += np.array(l.success)
        sent_len += np.array(l.sent_len)
        
    success /= n_sim
    sent_len /= n_sim
    return success, sent_len

def get_learning_time(learners):
    success, sent_len = get_success_and_length(learners)
    x_data = np.linspace(0,len(success),len(success))
    y_data = success
    popt,pcov = scipy.optimize.curve_fit(logistic,x_data,y_data,maxfev=10000)
    return 2*popt[-1]

def get_averaged_final_index(learners):
    final = 0
    for l in learners:
        final += l.final_index
        
    return final/len(learners)

    


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
Learner.beta = 1.
Learner.positive_reinforcement = 25.
Learner.negative_reinforcement = -10.

RWLearner.alpha = 0.1
RWLearner.beta = 1.
RWLearner.positive_reinforcement = 25.
RWLearner.negative_reinforcement = -10.

#############################################################
#
#       Initializing the learners
#
#############################################################

# number of simulations
n_sim = 100
n_trials = 3000



###############################################################
#
#       Definition of the language using a probabilistic CFG
#
###############################################################

n_verbs = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
n_nouns = [2]
len_sent = []

learning_times = []

for i in range(len(n_verbs)):

    print('Defining the grammar')

    # definition of the grammar
    # Vocabulary
    number_of_verbs = n_verbs[i]
    number_of_nouns = n_nouns[0]
    len_sent.append(number_of_nouns^2*number_of_verbs)


    verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]


    terminals2 = flatten([verbs,nouns])
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





#############################################################
#
#   Creating the stimuli stream
#
#############################################################
    print('Creating the stimuli stream')

# Create stimuli stream
    stimuli_stream = Raw_input(3*n_trials,cfgNVN)

    print('Initializing learners')

    # Select type of chunking mechanism
    #typ = 'right'
    typ = 'flexible'
    border = 'nxt'






# Create as many learners as number of simulations.
    learners = [RWLearner(n_trials = n_trials, border = border) for i in range(n_sim)]

#############################################################
#
#       Running the simulation in parallel
#
#############################################################    
    print('Running the simulation in parallel')

# # Run the simulation in parallel
    with cf.ThreadPoolExecutor() as executor:
        print("Number of worker threads:", executor._max_workers)
        results = [executor.submit(run_simulation, l) for l in learners]
    
        # Iterate over the results as they become available
        for future in cf.as_completed(results):
            result = future.result()
        # Combine the result with other results as necessary
        lt = get_learning_time(learners) 
        print('Learning time: '+str(lt))
        learning_times.append(lt)

    
#############################################################
#
#       Postprocessing
#
#############################################################
print('Postprocessing')
print([8,32,4*32,16*32])
sentences = [8,32,4*32,16*32]

#plt.plot(betas,learning_times)
plt.plot(n_verbs,np.array(learning_times)/np.array(len_sent))
plt.xlabel('Number of verbs')
plt.ylabel('learning time per sentence')
#plot_learning_curve(learners)
#plot_success_norm(learners)
#save_grammar_to_file(learners, 'testGrammarRC.xlsx')


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# lensent = []
# for i in range(n_sim):
#     lensent.append(len(learners[i].sentences))
    
# meanlen = np.mean(np.array(lensent))/7**3 
# print(meanlen)
#for s in learners[0].sentences:
#    print(s)