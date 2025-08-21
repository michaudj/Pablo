# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:22:47 2025

@author: jmd01
"""

from Learner import Learner,LearnerConfig
from Population import Population
from grammars import *
import matplotlib.pyplot as plt

# Things to try, only reinforce types for non-complex SChunk.
# Improve the assign_types function to do a greedy search especially in the case 1 0\0/0, the left zero should be retyped. Same on the right.
# Improve the decision support implementation. Currently very crude.
# Consider using the typatory to keep the number of types to a minimum.


n_trial = 30000
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
                       bad_type_threshold = -0.5,
                       good_type_threshold = 0., 
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
                       bad_type_threshold = -0.5,
                       good_type_threshold = 0., 
                       tau = 0.1,
                       type_on = False)



learner = Learner(config)

learner.learn(create_stimuliMD())
ma = learner.history.plot_moving_average(100)

learner_t = Learner(config_t)

learner_t.learn(create_stimuliMD())
ma_t = learner_t.history.plot_moving_average(100)


plt.figure(figsize=(10, 5))
plt.plot(ma_t, label='With types')
plt.plot(ma, label='Without types')
plt.xlabel('Trial')
plt.ylabel('Success rate')
plt.ylim((0,1))
plt.title('Learning Progress: NVN 20 verbs, 50 nouns')
plt.grid(True)
plt.legend()