# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:00:55 2023

@author: jmd01
"""


from newLearner import Learner
from new_raw_input import Raw_input, ProbabilisticGrammar
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as cf
from datetime import datetime
import pandas as pd
start_time = datetime.now()
# do your work here


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

def save_grammar_to_file(learners,filename,threshold=2):
    count_sent = {}

    for l in learners:
        gram = l.extract_sentences(threshold)
        #gram = l.sentences
        for s in gram:
            if s in count_sent:
                count_sent[s] += 1
            else:
                count_sent[s] = 1
    if len(set(gram))==0:
        print('No sentence identified yet...')
        return -1
                
    data = []          
    for s in set(gram):
        data.append({"Sentence": str(s.remove_structure()), "Parsing": str(s), "Count": count_sent[s]})

    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='Sentence', key=lambda x: x.str.len(),inplace=True)
    df_sorted = df.reindex(df_sorted)
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted.to_excel(filename)
    
def get_success_and_length(learners):
    success = 0.*np.array(learners[0].success)
    sent_len = 0.*np.array(learners[0].sent_len)
    for l in learners:
        success += np.array(l.success)
        sent_len += np.array(l.sent_len)
        
    success /= n_sim
    sent_len /= n_sim
    return success, sent_len
    
def plot_learning_curve(learners):
    # Process the result    
    success, sent_len = get_success_and_length(learners)

    trial_vec = range(learners[0].n_trials+1)

    # Plot the results
    plt.scatter(trial_vec,success,s = 1,c=sent_len)
    plt.xlabel('# of reinforcement')
    plt.ylabel('frequency of correct identifications')
    plt.colorbar()
    plt.show()
    
def success_by_length(learners):
    filtered_success = {}
    length_filtered = {}
    for l in learners:
        filtered_success[l] = {length: np.array([result if length_ == length else 0 for result, length_ in zip(l.success, l.sent_len)]) for length in set(l.sent_len)}
        length_filtered[l] = {length: np.array([1 if length_ == length else 0 for result, length_ in zip(l.success, l.sent_len)]) for length in set(l.sent_len)}
    count = {}
    successes = {}
    for i in learners:
        for leng in filtered_success[learners[0]]:
            if leng not in count:
                count[leng]=0*filtered_success[learners[0]][leng]
            count[leng]+=length_filtered[i][leng]
            if leng not in successes:
                successes[leng] = 0*filtered_success[learners[0]][leng]
            successes[leng] += filtered_success[i][leng]
            


    successes_norm = {}
    for leng in successes:
        successes_norm[leng] = successes[leng]/count[leng]

        
    
    return successes_norm

def plot_success_norm(learners):
    successes_norm = success_by_length(learners)
    for key in successes_norm:
        plt.plot(successes_norm[key], '.', markersize=1)
    plt.show()
    


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


###############################################################
#
#       NVN + relative clauses with relative pronouns language
#
###############################################################

# Grammatical rules
production_rulesRCP = {
    'S': [['N', 'VP'],['N','VP','RelCl'],['N','VP','RelCl','RelCl']],
    'VP': [['V','N'],['V']],
    'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsRCP = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [.5, .5],
    'RelCl': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

relweight = np.array([1/2**i for i in range(len(relpron))])
relweight /= np.sum(relweight)

nweight = np.array([1/2**i for i in range(len(nouns))])
nweight /= np.sum(nweight)

vweight = np.array([1/2**i for i in range(len(verbs))])
vweight /= np.sum(vweight)

weightsRCPZipf = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [.5, .5],
    'RelCl': relweight,
    'N': nweight,
    'V': vweight   
    }

# Context free grammar
cfgRCP = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesRCP,weightsRCPZipf)


###############################################################
#
#       NVN + relative clauses without relative pronouns language
#
###############################################################

# Grammatical rules
production_rulesRC = {
    'S': [['N', 'VP'],['N','VP','RelCl'],['N','VP','RelCl','RelCl']],
    'VP': [['V','N']],
    'RelCl': [['VP'] ],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsRC = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [1],
    'RelCl': [1],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

# Context free grammar
cfgRC = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesRC,weightsRC)

#############################################################
#
#       Yang and Piantadosi grammar
#
#############################################################

number_of_verbs = 1
number_of_nouns = 1
number_of_adj = 1
number_of_relpron = 1
number_of_det = 1
number_of_prep = 1

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]
prep = ['p' + str(i) for i in range(1, number_of_prep+1)]


terminalsYP = flatten([verbs,nouns,adjs,relpron,det,prep])
non_terminalsYP = ['S','NP','VP','AP','PP','N','V','A','D','P','rel']

production_rulesYP = {
    'S': [['NP', 'VP']],
    'NP': [['N'],['D','N'],['D','AP','N'],['NP','PP']],
    'VP': [['V'],['V','NP'],['V','rel','S'],['VP','PP']],
    'AP': [['A'],['A','AP' ] ],
    'PP': [['P','NP']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
    'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
    'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
    'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
    'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)]
}

weightsYP = {
    'S': [1.0],
    'NP': [.25,.25,.25,.25],
    'VP': [.25,.25,.25,.25],
    'AP': [.5,.5 ],
    'PP': [1],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
    'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
    'D': [1/number_of_det for i in range(1, number_of_det+1)],
    'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
    'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)]  
    }

cfgYP = ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)

#############################################################
#
#       Initializing the learners
#
#############################################################

# number of simulations
n_sim = 10
n_trials = 50000

#############################################################
#
#   Creating the stimuli stream
#
#############################################################
print('Creating the stimuli stream')

# Create stimuli stream
stimuli_stream = Raw_input(3*n_trials,cfgYP)

print('Initializing learners')

# Select type of chunking mechanism
#typ = 'right'
typ = 'flexible'
border = 'net'

# Create as many learners as number of simulations.
learners = [Learner(n_trials = n_trials, t=typ, border = border) for i in range(n_sim)]

# print('Running simulation')
# count = 0
# for l in learners:

#     count += 1
#     print(count)
#     l.learn(stimuli_stream)

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

    
#############################################################
#
#       Postprocessing
#
#############################################################
print('Postprocessing')


plot_learning_curve(learners)
#plot_success_norm(learners)
save_grammar_to_file(learners, 'testGrammarRC.xlsx')


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

