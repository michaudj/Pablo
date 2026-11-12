# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:22:47 2025

@author: jmd01
"""
import sys
from contextlib import redirect_stdout

from Learner import Learner,LearnerConfig
from RawInput import RawInput2
#from Population import Population
from grammar_definitions import *
#import matplotlib.pyplot as plt







# Things to try, only reinforce types for non-complex SChunk.
# Improve the assign_types function to do a greedy search especially in the case 1 0\0/0, the left zero should be retyped. Same on the right.
# Improve the decision support implementation. Currently very crude.
# Consider using the typatory to keep the number of types to a minimum.


n_trial = 180000
alpha = 0.1
alpha_v = 1
beta = 1.
RW = True

config_t = LearnerConfig(n_trials=n_trial,
                       border = 'cont',
                       initial_value_chunking=-1.,
                       initial_value_border=1.,
                       alpha= alpha,
                       alpha_v=alpha_v,
                       beta= beta,
                       positive_reinforcement = 25,
                       negative_reinforcement = -10,
                       RW=RW,
                       chaining = False,
                       bad_type_threshold = -1.1,
                       good_type_threshold = 2., 
                       tau = 0.1,
                       type_on = True)

config = LearnerConfig(n_trials=n_trial,
                       border = 'cont',
                       initial_value_chunking=-1.,
                       initial_value_border=1.,
                       alpha= alpha,
                       alpha_v=alpha_v,
                       beta= beta,
                       positive_reinforcement = 25,
                       negative_reinforcement = -10,
                       RW=RW,
                       chaining = False,
                       bad_type_threshold = 0.,
                       good_type_threshold = 0., 
                       tau = 0.1,
                       type_on = False)





list_of_paths =['MD_1-2adj_n40a8mv14dv2','MD_1adj_n40a8mv14dv2','MDn40a8mv14dv2',
                'NVN_1-2adj_n40a8v16','NVN_1adj_n40a8v16','NVN_n40a8v16',
                'NVN_n200v40','rel_1-2adj_n40a8rp2mv14dv2','rel_1adj_n40a8rp2mv14dv2',
                'rel_n40rp2mv14dv2']

for ind in [6]:
    print(list_of_paths[ind])
    stimuli = RawInput2.from_csv('grammars/'+list_of_paths[ind]+'/'+list_of_paths[ind]+'_dataset/train.csv')
    #stimili = RawInput2.from_grammar(grammar, 1_000_000)
    stimuli_test = RawInput2.from_csv('grammars/'+list_of_paths[ind]+'/'+list_of_paths[ind]+'_dataset/test.csv')
    
    learner_t_rel = Learner(config_t)
    with open(list_of_paths[ind]+"train.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            learner_t_rel.learn(stimuli)
    ma_t_rel = learner_t_rel.history.plot_moving_average(50)
    #smoothed_t = learner_t_rel.history.gaussian_smooth_uneven_fast(1500)
    
    #learner_rel = Learner(config)
    
    #learner_rel.learn(stimuli)
    #ma_rel = learner_rel.history.plot_moving_average(50)
    #smoothed = learner_rel.history.gaussian_smooth_uneven_fast(1500)
    
    print('testing')
    
    learner_t_rel.history.reset()
    with open(list_of_paths[ind]+"test.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            learner_t_rel.test(stimuli_test,1000)
    ma_t_test = learner_t_rel.history.plot_moving_average(50)
    print(np.mean(learner_t_rel.history.success))
    print(np.std(learner_t_rel.history.success))

