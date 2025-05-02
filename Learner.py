# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:18:18 2025

@author: jmd01
"""

from copy import deepcopy
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List


from SChunk import SChunk, ChunkPair

import sys
sys.setrecursionlimit(1500)


@dataclass
class LearnerConfig:
    n_trials: int= 10
    border: str = 'next'
    initial_value_chunking: float = -1.
    initial_value_border: float = 1.
    alpha: float = 0.2
    beta: float = 1.
    positive_reinforcement: float = 5.
    negative_reinforcement: float = -10.
    RW: bool = False
    # parameter for choosing type of learner (RW or not.)

class LongTermMemory():
    
    def __init__(self,config):
        self.initial_value_chunking = config.initial_value_chunking
        self.initial_value_border = config.initial_value_border
        self.behaviour_repertoire = {} # dictionary of where the keys are couples of chunks and the value a list of behavioural values

    
    def add(self,couple):
        if couple not in self.behaviour_repertoire:
            values = [self.initial_value_border] + [self.initial_value_chunking] * (couple.s1.depth + 1)
            
            self.behaviour_repertoire[couple] =np.array(values)
         
    def update_repertoire(self,couple): # couple must be a couple of SChunks
        subcouples = couple.get_sub_couples()
        for c in subcouples:
            self.add(c)
            


@dataclass
class LearningHistory:
    success: List[int] = field(default_factory=list)
    sent_len: List[int] = field(default_factory=list)

    def record(self, success_value: int, length: int):
        self.success.append(success_value)
        self.sent_len.append(length)


            
class WorkingMemory():
    
    def __init__(self,learner,config: LearnerConfig):
        self.learner = learner
        self.reinforcer = Reinforcer(learner,config)
        self.events = []
        self.border_before = True
        self.border_within = False
        self.border_type = config.border
        self.beta = config.beta
        
    def place_border_and_reinforce(self,stimuli_stream,s2,s2_index, reinforcement = True):
        self.learner.n_reinf += 1
        sent_length = stimuli_stream.length_current_sent(s2_index - 1)
        is_border = stimuli_stream.border_before[s2_index]
        if is_border and not self.border_within and self.border_before:
            # Good unit
            if reinforcement:
                self.reinforcer.reinforce(self.events,reinforcement = 'positive')  
            self.learner.history.record(1,sent_length)
        else:
            # Bad unit
            if reinforcement:
                self.reinforcer.reinforce(self.events,reinforcement = 'negative') 
            self.learner.history.record(0,sent_length)
            
        if self.border_type == 'next':
            new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
            new_s1 = SChunk(new_s1)
        else:
            self.border_before = stimuli_stream.border_before[s2_index]
            new_s1,s2_index = s2, s2_index + 1

        self.border_within = False
        
        self.events = []
        
        return new_s1, s2_index

    def chunk(self,pair,response,stimuli_stream,s2_index):
        if not self.border_within:
            self.border_within = stimuli_stream.border_before[s2_index]
         
        # Perform chunking at correct level
        new_s1 = pair.s1.chunk_at_depth(pair.s2,depth=pair.s1.depth+1-response) 
        s2_index+=1 
        return new_s1, s2_index
    
    def respond(self,stimuli_stream,s1,s2_index,reinforcement = True):
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.read_stimuli(s2_index))
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
        
        pair = ChunkPair((s1,s2))
        self.learner.ltm.update_repertoire(pair)

        response = self.choose_behaviour(pair)

        self.events.append((pair,response))
         
        if response == 0: # boundary placement
            new_s1, s2_index = self.place_border_and_reinforce(stimuli_stream, s2, s2_index, reinforcement=reinforcement)
               
        else: # some type of chunking occurs
            # Check if there was a border
            new_s1, s2_index = self.chunk(pair, response, stimuli_stream,s2_index)     
        return new_s1, s2_index
    
    
    def choose_behaviour(self,couple):
        b_range = len(self.learner.ltm.behaviour_repertoire[couple])
        z = self.Q_tilde(couple,b_range)
        weights = np.exp(self.beta * z)
        options = [i for i in range(b_range)]
        response = random.choices(options,weights/np.sum(weights))
        return response[0]  
    

    def Q_tilde(self,couple,b_range):
        z = deepcopy(self.learner.ltm.behaviour_repertoire[couple])
        subpairs = couple.get_sub_couples()
        
        norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
        # Accumulate support from subchunks
        for pair in subpairs:
            lenp = len(self.learner.ltm.behaviour_repertoire[pair])
            z[:lenp] += self.learner.ltm.behaviour_repertoire[pair]
        # Take the average
        z /= norm_vec
        return z 
            
class Reinforcer():
    
    def __init__(self,learner,config: LearnerConfig):
        self.alpha = config.alpha
        self.positive_reinforcement = config.positive_reinforcement
        self.negative_reinforcement = config.negative_reinforcement
        self.learner = learner
        self.RW = config.RW
        
    def __repr__(self):
        return f"Reinforcer of {self.learner}"
    
    def get_sub_events(self,event):
        couple, r = event
        subevents = []

        #subevents.append((couple,r))
        for subcouple in couple.get_sub_couples():
            if r < len(self.learner.ltm.behaviour_repertoire[subcouple]):
                subevents.append((subcouple,r))        
        return subevents
            
    def get_sub_eventsRW(self,event):
        couple, r = event
        subevents = []
        
        Q = self.learner.ltm.behaviour_repertoire[couple][r]
        #subevents.append((couple,r))
        for subcouple in couple.get_sub_couples():
            if r < len(self.learner.ltm.behaviour_repertoire[subcouple]):
                subevents.append((subcouple,r)) 
                Q += self.learner.ltm.behaviour_repertoire[subcouple][r]
        return subevents, Q
        
        
    def reinforce(self,events, reinforcement = 'positive'):
        if reinforcement == 'positive':
            u = self.positive_reinforcement
        elif reinforcement == 'negative':
            u = self.negative_reinforcement

        for couple,r in events:
            if self.RW:
                subevents, Q = self.get_sub_eventsRW((couple,r))
                
                for p,rr in subevents:
                    self.learner.ltm.behaviour_repertoire[p][rr]+= self.alpha * (u - Q)
            else:
                subevents=self.get_sub_events((couple,r))
    
                for p,rr in subevents:
                    self.learner.ltm.behaviour_repertoire[p][rr] += self.alpha * (u - self.learner.ltm.behaviour_repertoire[p][rr])



class Learner():
    ID = 0
    
    def __init__(self, config: LearnerConfig): #Use a config dataclass
        self.ID = Learner.ID + 1
        Learner.ID +=1
        
        self.ltm = LongTermMemory(config)
        self.wm = WorkingMemory(self,config) # alpha, 
        self.history = LearningHistory()
        
        
        # self.border_type = border # or 'default'
        self.n_trials = config.n_trials
        self.n_reinf = 0

        
        self.final_index = 0
        
        
        
    def __repr__(self):
        return f'Learner {self.ID}'
       
    def learn(self,stimuli_stream):
        # initialize stimuli
        s1 = SChunk(stimuli_stream.read_stimuli(0))
        s2_index = 1
        #for t in range(self.n_trials):
        while self.n_reinf <= self.n_trials:
            s1, s2_index = self.wm.respond(stimuli_stream, s1, s2_index)

        self.final_index = s2_index
        
        