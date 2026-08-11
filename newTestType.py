# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:22:47 2025

@author: jmd01
"""

from Learner import Learner,LearnerConfig
from RawInput import RawInput2
#from Population import Population
from grammar_definitions import *
#import matplotlib.pyplot as plt







# Things to try, only reinforce types for non-complex SChunk.
# Improve the assign_types function to do a greedy search especially in the case 1 0\0/0, the left zero should be retyped. Same on the right.
# Improve the decision support implementation. Currently very crude.
# Consider using the typatory to keep the number of types to a minimum.


n_trial = 100000
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
                       good_type_threshold = 3., 
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



learner_t_rel = Learner(config_t)

rel_grammar = create_grammar_rel()
stimuli = RawInput2.from_grammar(rel_grammar,1_000_000)

learner_t_rel.learn(stimuli)
ma_t_rel = learner_t_rel.history.plot_moving_average(500)
smoothed_t = learner_t_rel.history.gaussian_smooth_uneven_fast(1500)

learner_rel = Learner(config)

learner_rel.learn(stimuli)
ma_rel = learner_rel.history.plot_moving_average(500)
smoothed = learner_rel.history.gaussian_smooth_uneven_fast(1500)

#print('testing')

#learner_t_rel.history.reset()
#learner_t_rel.test(create_stimuli_rel_test(),1000)
#ma_t_test = learner_t_rel.history.plot_moving_average(100)

#learner_rel.history.reset()
#learner_rel.test(create_stimuli_rel_test(),1000)
#ma_test = learner_rel.history.plot_moving_average(100)

# learner_t.ltm.display_typings_of_elements()


# plt.figure(figsize=(10, 5))
# plt.plot(ma_NVN, label='Model 1')
# plt.plot(ma_t_NVN, label='Model 2')

# plt.xlabel('Episode')
# plt.ylabel('Success rate')
# plt.ylim((0,1))
# plt.title('Learning Progress: NVN language: 50 nouns and 20 verbs')
# plt.grid(True)
# plt.legend()
# plt.savefig('NVN.pdf',format='pdf')

# plt.figure(figsize=(10, 5))
# plt.plot(ma_MD, label='Model 1')
# plt.plot(ma_t_MD, label='Model 2')

# plt.xlabel('Episode')
# plt.ylabel('Success rate')
# plt.ylim((0,1))
# plt.title('Learning Progress: MD language 20 mono, 5 ditransitive verbs, 40 nouns')
# plt.grid(True)
# plt.legend()
# plt.savefig('MD.pdf',format='pdf')

# plt.figure(figsize=(10, 5))
# plt.plot(ma, label='Model 1')
# plt.plot(ma_t, label='Model 2')

# plt.xlabel('Episode')
# plt.ylabel('Success rate')
# plt.ylim((0,1))
# plt.title('Learning Progress: Rel language 20 mono, 5 ditransitive verbs, 40 nouns, 1 rel')
# plt.grid(True)
# plt.legend()
# plt.savefig('Rel.pdf',format='pdf')

# # sys.stdout = orig_stdout