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
import pandas as pd
from TypeNew import Type, TChunk

import math


from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.styles import PatternFill

import matplotlib.pyplot as plt

from typing import Tuple, Dict, Any, Optional



from SChunk import SChunk, ChunkPair

import sys
sys.setrecursionlimit(1500)



def softmax(weights: Dict, tau: float = 1.0) -> Dict:
    """Compute softmax distribution with temperature tau."""
    if not weights:
        return {}
    keys, values = zip(*weights.items())
    scaled = [v / tau for v in values]
    max_scaled = max(scaled)  # for numerical stability
    exp_values = [math.exp(s - max_scaled) for s in scaled]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]
    return dict(zip(keys, probs))

def merged_softmax_choice(
    left: Dict, right: Dict, tau: float = 1.0
) -> Tuple[str, Optional[object]]:
    """
    Returns (side, selected_key) where side ∈ {'left', 'right'}.
    Handles empty dicts gracefully.
    """
    if not left and not right:
        return random.choice([('left', None), ('right', None)])

    if not left:
        chosen = softmax_choice(right, tau)
        return 'right', chosen
    if not right:
        chosen = softmax_choice(left, tau)
        return 'left', chosen

    # Both non-empty: proceed with merged softmax
    left_soft = softmax(left, tau)
    right_soft = softmax(right, tau)

    merged_keys = list(left_soft.keys()) + list(right_soft.keys())
    merged_probs = list(left_soft.values()) + list(right_soft.values())

    chosen_key = random.choices(merged_keys, weights=merged_probs, k=1)[0]
    if chosen_key in left_soft:
        return 'left', chosen_key
    else:
        return 'right', chosen_key

def softmax_choice(weights: Dict, tau: float = 1.0):
    """Sample a key from a softmax distribution of weights."""
    probs = softmax(weights, tau)
    keys, values = zip(*probs.items())
    return random.choices(keys, weights=values, k=1)[0]



@dataclass
class LearnerConfig:
    n_trials: int= 10
    border: str = 'next'
    initial_value_chunking: float = -1.
    initial_value_border: float = 1.
    alpha: float = 0.2
    alpha_v: float = 0.3
    beta: float = 1.
    positive_reinforcement: float = 5.
    negative_reinforcement: float = -10.
    RW: bool = False
    chaining: bool = False
    bad_type_threshold: float = -1
    good_type_threshold: float = 0.1
    tau: float = 2.0 # softmax parameter of TypeAssigner
    # parameter for choosing type of learner (RW or not.)

class LongTermMemory():
    
    def __init__(self,config):
        self.initial_value_chunking = config.initial_value_chunking
        self.initial_value_border = config.initial_value_border
        self.behaviour_repertoire = dict() # dictionary of where the keys are couples of chunks and the value a list of behavioural values
        self.chunk_values = dict()
        
        self.chunk_type_associations = dict()
        self.typatory = dict()

    
    def add(self,couple):
        if couple not in self.behaviour_repertoire:
            values = [self.initial_value_border] + [self.initial_value_chunking] * (couple.s1.depth + 1)
            
            self.behaviour_repertoire[couple] =np.array(values)
         
    def update_repertoire(self,couple): # couple must be a couple of SChunks
        subcouples = couple.get_sub_couples()
        for c in subcouples:
            self.add(c)
            
    def update_chunk(self,chunk):
        if chunk not in self.chunk_values:
            self.chunk_values[chunk] = 0.0
            
    def update_typatory(self,ttype):
        if ttype not in self.typatory:
            self.typatory[ttype] = 0.0
            
    def update_chunk_type_associations(self,chunk,ttype):
        if chunk not in self.chunk_type_associations:
            self.chunk_type_associations[chunk] = dict()
        if ttype not in self.chunk_type_associations[chunk]:
            self.chunk_type_associations[chunk][ttype] = 0.0
            



    def write_behaviour_repertoire_to_xlsx(self, filename="structured_output.xlsx"):
        # Extract key attributes and values
        rows = []
        for key, values in self.behaviour_repertoire.items():
            for i, value in enumerate(values):
                rows.append({
                    's1': repr(key.s1),
                    's2': repr(key.s2),
                    'length': len(key.s1),
                    'index': i,
                    'value': value
                })
    
        # Create DataFrame
        df = pd.DataFrame(rows)
    
        # Pivot to wide format
        df_pivoted = df.pivot_table(
            index=['length', 's1', 's2'],
            columns='index',
            values='value',
            aggfunc='first'
        ).reset_index()
    
        # Sort by length, s1, s2
        df_pivoted = df_pivoted.sort_values(by=['length', 's1', 's2'])
    
        # Write to Excel
        df_pivoted.to_excel(filename, index=False)
    
        # Post-process Excel file with openpyxl
        wb = load_workbook(filename)
        ws = wb.active
    
        # Bold headers
        for cell in ws[1]:
            cell.font = Font(bold=True)
    
        # Freeze top row
        ws.freeze_panes = 'A2'
        
    
        # Define fill color for max value
        highlight_fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")  # Gold
        
        # Find the first data row (assumed row 2)
        first_data_row = 2
        last_row = ws.max_row
        first_data_col = 4  # Adjust if your "value" columns start later
        last_col = ws.max_column
        
        # Iterate through rows and highlight the max value
        for row in ws.iter_rows(min_row=first_data_row, max_row=last_row,
                                min_col=first_data_col, max_col=last_col):
            # Extract values and find max
            values = [cell.value for cell in row if isinstance(cell.value, (int, float))]
            if not values:
                continue
            max_val = max(values)
        
            # Highlight all cells with the max value
            for cell in row:
                if cell.value == max_val:
                    cell.fill = highlight_fill

    
        wb.save(filename)

            
@dataclass
class LearningHistory:
    success: List[int] = field(default_factory=list)
    sent_len: List[int] = field(default_factory=list)

    def record(self, success_value: int, length: int):
        self.success.append(success_value)
        self.sent_len.append(length)
        


    def plot_moving_average(self, window_size=10, show=True, save_path=None):
        if len(self.success) < window_size:
            print(f"Not enough data to compute moving average (need at least {window_size}).")
            return

        ma = [sum(self.success[i:i+window_size]) / window_size 
              for i in range(len(self.success) - window_size + 1)]

        plt.figure(figsize=(10, 5))
        plt.plot(ma, label=f'{window_size}-trial moving avg')
        plt.xlabel('Trial')
        plt.ylabel('Success rate')
        plt.title('Learning Progress')
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        

    
    def plot_moving_average_by_length_timed(self, window_size=10, show=True, save_path=None, lengths=None):
        data = pd.DataFrame({
            'success': self.success,
            'length': self.sent_len
        })
    
        if lengths is None:
            lengths = sorted(data['length'].unique())
    
        plt.figure(figsize=(10, 5))
        learning_curves = {}
    
        for length in lengths:
            # Create a time-aligned series: NaN for trials not matching the length
            mask = data['length'] == length
            successes = data['success'].where(mask)
    
            # Compute moving average ignoring NaNs
            ma = successes.rolling(window=window_size, min_periods=1).mean()
    
            plt.plot(ma, label=f'len={length}')
            
            # Store the aligned success list (not the moving avg) for averaging across learners
            learning_curves[length] = list(successes)
    
        plt.xlabel('Trial (global time)')
        plt.ylabel('Success rate')
        plt.title(f'Learning Progress by Sentence Length (Aligned in Time, window={window_size})')
        plt.grid(True)
        plt.legend()
    
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        
        return learning_curves

            
class WorkingMemory():
    
    def __init__(self,learner,config: LearnerConfig):
        self.learner = learner
        self.reinforcer = Reinforcer(learner,config)
        self.type_assigner = TypeAssigner(learner,config)
        self.events = []
        self.typing_events = dict()
        self.ts1_list = Type.EMPTY
        self.typing_used = False
        self.border_before = True
        self.border_within = False
        self.border_type = config.border
        self.beta = config.beta
        self.pos = config.positive_reinforcement
        self.neg = config.negative_reinforcement
    
    def respond(self,stimuli_stream,s1,s2_index,reinforcement = True):
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.read_stimuli(s2_index))
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
        
        pair = ChunkPair((s1,s2))
        # Assign types
        # Update memory
        self.learner.ltm.update_repertoire(pair)

        response = self.choose_behaviour(pair)

        self.events.append((pair,response))
        
        if response == 0: # boundary placement
            self.learner.n_reinf += 1
            sent_length = stimuli_stream.length_current_sent(s2_index - 1)
            if self.is_border_correct(stimuli_stream,s2_index):
                reward = self.pos
                self.learner.history.record(1,sent_length)
                if reinforcement:
                    self.reinforcer.reinforce2(self.events,reward)
                    #self.reinforcer.reinforce_value_hierarchical(pair.s1,reward)
            else:
                reward = self.neg
                self.learner.history.record(0,sent_length)
                if reinforcement:
                    self.reinforcer.reinforce2(self.events,reward)
                    #self.reinforcer.reinforce_value(pair.s1,reward)
            
            new_s1, s2_index = self.get_new_s1(stimuli_stream,s2_index, s2)
            
            self.events = []
               
        else: # some type of chunking occurs
            new_s1, s2_index = self.chunk(pair, response, stimuli_stream,s2_index) 
  
        return new_s1, s2_index
    
    def respond_with_type(self,stimuli_stream,s1,s2_index,reinforcement = True):
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.read_stimuli(s2_index))
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
        
        pair = ChunkPair((s1,s2))
        (self.ts1, ts2) = self.type_assigner.assign_type(pair,self.ts1) # self.ts1 is a TChunk, while ts2 is a Type object 
        # Assign types
        # Update memory
        self.learner.ltm.update_repertoire(pair)
        
        response = self.choose_behaviour(pair)

        #response = self.choose_behaviour_with_types(pair, (self.ts1,ts2)) # Set also whether self.typing_used is True or False

        self.events.append((pair,response))
        
        
        if response == 0: # boundary placement
            self.learner.n_reinf += 1
            sent_length = stimuli_stream.length_current_sent(s2_index - 1)
            
            if self.is_border_correct(stimuli_stream,s2_index):
                reward = self.pos
                self.learner.history.record(1,sent_length)
                if not self.typing_used:
                   self.type_assigner.type_sentence(pair.s1)
                if reinforcement:
                    self.reinforcer.reinforce2(self.events,reward)
                    self.reinforcer.reinforce_types(self.typing_events,reward)
            else:
                reward = self.neg
                self.learner.history.record(0,sent_length)
                if reinforcement:
                    self.reinforcer.reinforce2(self.events,reward)
                    #if self.typing_used:
                    #   self.typing_events = self.extract_typing_events()
                    #   self.reinforcer.reinforce_types(self.typing_events,reward)
            
            new_s1, s2_index = self.get_new_s1(stimuli_stream,s2_index, s2)
            
            self.events = []
            self.typing_events = []
               
        else: # some type of chunking occurs
            new_s1, s2_index = self.chunk(pair, response, stimuli_stream,s2_index) 
  
        return new_s1, s2_index
    
    def is_border_correct(self,stimuli_stream,s2_index):
        is_border = stimuli_stream.border_before[s2_index]
        return is_border and not self.border_within and self.border_before
    
    def chunk(self,pair,response,stimuli_stream,s2_index):
        if not self.border_within:
            self.border_within = stimuli_stream.border_before[s2_index]
         
        # Perform chunking at correct level
        new_s1 = pair.s1.chunk_at_depth(pair.s2,depth=pair.s1.depth+1-response) 
        s2_index+=1 
        return new_s1, s2_index
    
    def get_new_s1(self,stimuli_stream,s2_index,s2):
        if self.border_type == 'next':
            new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
            new_s1 = SChunk(new_s1)
        else:
            self.border_before = stimuli_stream.border_before[s2_index]
            new_s1,s2_index = s2, s2_index + 1

        self.border_within = False
        return new_s1, s2_index
        
    def extract_typing_events(self):
        # Use the structure of self.ts1 to extract the typing events
        pass
    
    def respond_with_chaining(self,stimuli_stream,s1,s2_index,reinforcement = True):
        # Positive and negative propagation to chunks
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.read_stimuli(s2_index))
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
        
        pair = ChunkPair((s1,s2))
        self.learner.ltm.update_repertoire(pair)

        response = self.choose_behaviour(pair)

        event = [(pair,response)]
         
        if response == 0: # boundary placement
            self.learner.n_reinf += 1
            sent_length = stimuli_stream.length_current_sent(s2_index - 1)
            if self.is_border_correct(stimuli_stream,s2_index):
                reward = self.pos
                self.learner.history.record(1,sent_length)
            else:
                reward = self.neg
                self.learner.history.record(0,sent_length)
            
            new_s1, s2_index = self.get_new_s1(stimuli_stream,s2_index, s2)
               
        else: # some type of chunking occurs
            # Check if there was a border
            new_s1, s2_index = self.chunk(pair, response, stimuli_stream,s2_index) 
            self.learner.ltm.update_chunk(new_s1)
            reward = self.learner.ltm.chunk_values[new_s1]
            
        # Reinforce the event and the value of s1.
        if reinforcement:
            self.reinforcer.reinforce2(event,reward)
            self.reinforcer.reinforce_value(pair.s1,reward)
            
        return new_s1, s2_index
    
    def respond_with_chaining2(self,stimuli_stream,s1,s2_index,reinforcement = True):
        # Only positive propagation to chunks
        
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.read_stimuli(s2_index))
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
        
        pair = ChunkPair((s1,s2))
        self.learner.ltm.update_repertoire(pair)

        response = self.choose_behaviour(pair)

        event = [(pair,response)]
         
        if response == 0: # boundary placement
            self.learner.n_reinf += 1
            sent_length = stimuli_stream.length_current_sent(s2_index - 1)
            if self.is_border_correct(stimuli_stream,s2_index):
                reward = self.pos
                self.learner.history.record(1,sent_length)
                if reinforcement:
                    self.reinforcer.reinforce2(event,reward)
                    self.reinforcer.reinforce_value_hierarchical(pair.s1,reward)
            else:
                reward = self.neg
                self.learner.history.record(0,sent_length)
                if reinforcement:
                    self.reinforcer.reinforce2(event,reward)
                    #self.reinforcer.reinforce_value(pair.s1,reward)
            
            new_s1, s2_index = self.get_new_s1(stimuli_stream,s2_index, s2)
               
        else: # some type of chunking occurs
            # Check if there was a border
            new_s1, s2_index = self.chunk(pair, response, stimuli_stream,s2_index) 
            self.learner.ltm.update_chunk(new_s1)
            reward = self.learner.ltm.chunk_values[new_s1]
            if reinforcement:
                self.reinforcer.reinforce2(event,reward)
                self.reinforcer.reinforce_value_hierarchical(pair.s1,reward)
            
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

    def choose_behaviour_with_types(self, pair, types_pair):
        pass
            
class Reinforcer():
    
    def __init__(self,learner,config: LearnerConfig):
        self.alpha = config.alpha
        self.alpha_v = config.alpha_v
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
                    
    def reinforce2(self,events, reward):
        u = reward

        for couple,r in events:
            if self.RW:
                subevents, Q = self.get_sub_eventsRW((couple,r))
                
                for p,rr in subevents:
                    self.learner.ltm.behaviour_repertoire[p][rr]+= self.alpha * (u - Q)
            else:
                subevents=self.get_sub_events((couple,r))
    
                for p,rr in subevents:
                    self.learner.ltm.behaviour_repertoire[p][rr] += self.alpha * (u - self.learner.ltm.behaviour_repertoire[p][rr])

    def reinforce_value(self,chunk,reward):
        self.learner.ltm.update_chunk(chunk)
        self.learner.ltm.chunk_values[chunk] += self.alpha_v * (reward - self.learner.ltm.chunk_values[chunk])
        
    def reinforce_value_hierarchical(self,chunk,reward):
        chunks_list = [chunk]
        try:
            subchunks = chunk.get_right_subchunks2(0)
            for s in subchunks:
                chunks_list.append(s)
        except ValueError:
            pass 
        
        for c in chunks_list:
            self.learner.ltm.update_chunk(c)
            self.learner.ltm.chunk_values[c] += self.alpha_v * (reward - self.learner.ltm.chunk_values[c])
            
    def reinforce_types(self,typing_events,reward):
        for chunk,typ in typing_events.items():
            self.learner.ltm.update_chunk_type_associations(chunk,typ)
            self.learner.ltm.chunk_type_associations[chunk][typ] += self.alpha * (reward - self.learner.ltm.chunk_type_associations[chunk][typ])

class TypeAssigner():
    
    def __init__(self, learner, config: LearnerConfig):
        self.learner = learner
        self.bad_type_threshold = config.bad_type_threshold
        self.good_type_threshold = config.good_type_threshold
        self.tau = config.tau
        pass
    
    def assign_type(self, pair: ChunkPair):
        # Do type assignment taking into account the values associated to chunk and types
        # update longterm memory
        return (Type.EMPTY, Type.EMPTY)
    
    def choose_types(self, typ, s1, s2):
        def _compatible_pair(typ, left_candidates, right_candidates):
            # Tries to find a compatible pair
            chosen_pair = None
            for lt in left_candidates:
                for rt in right_candidates:
                    try:
                        if lt + rt == typ:
                            chosen_pair = (lt, rt)
                            break
                    except TypeError:
                        continue
                if chosen_pair:
                    break
            return chosen_pair
        
        def _dominant_type(typ, left_candidates, right_candidates,s1,s2):
            side, chosen = merged_softmax_choice(left_candidates, right_candidates, tau=self.tau)

            if chosen is None:
                # Fall back to a default random split
                left_type, right_type =typ.split(pu=0.5, 
                                                       prim='New',
                                                       bad_s1= self.extract_bad_types(s1),
                                                       bad_s2=self.extract_bad_types(s2))
                return (left_type, right_type)
            elif side == 'left':
                left_type, right_type = typ.split(pu=1.0, prim=chosen,bad_s2 = self.extract_bad_types(s2))
                return (left_type, right_type)
            else:
                left_type, right_type = typ.split(pu=0.0, prim=chosen,bad_s1 = self.extract_bad_types(s1))
                return (left_type, right_type)

        
        left_candidates = self.extract_good_types(s1)
        right_candidates = self.extract_good_types(s2)
        
        chosen_pair = _compatible_pair(typ,left_candidates,right_candidates)
        
            
        if not chosen_pair: # No compatible pairs have been found
                # Choose randomly a right of a left type that is good and construct the corresponding type on the other side
            chosen_pair = _dominant_type(typ,left_candidates,right_candidates,s1,s2)
            
        return chosen_pair
    
    def propagate_types(self,
                        chunk: SChunk,
                        current_type: Type) -> None:
        
        self.learner.wm.typing_events[chunk] = current_type
    
        if not isinstance(chunk.structure, list):
            return  # Base case: it's a leaf
        
        # Propagate to children
        left_chunk = chunk.get_left()
        right_chunk = chunk.get_right()
        
        
        left_type, right_type = self.choose_types(current_type,left_chunk,right_chunk)
        
        self.propagate_types(left_chunk, left_type)
        self.propagate_types(right_chunk, right_type)

    
    def type_sentence(self, s1: SChunk):
        typ = Type.SENTENCE # start with the sentence type
        # Get good and bad types for the components of s1 if its structure is complex
        self.learner.wm.typing_events = {}
        self.propagate_types(s1,typ)
        

    
    def extract_bad_types(self, chunk: SChunk):
        def filter_dict_below_threshold(data,threshold):
            result = {k: v for k, v in data.items() if v < threshold}
            return result if result else {}
        
        if chunk in self.learner.ltm.chunk_type_associations:
            return filter_dict_below_threshold(self.learner.ltm.chunk_type_associations[chunk],self.bad_type_threshold)
        else:
            return {}
        
    def extract_good_types(self, chunk: SChunk):
        def filter_dict_above_threshold(data,threshold):
            result = {k: v for k, v in data.items() if v > threshold}
            return result if result else {}
        
        if chunk in self.learner.ltm.chunk_type_associations:
            return filter_dict_above_threshold(self.learner.ltm.chunk_type_associations[chunk],self.good_type_threshold)
        else:
            return {}




class Learner():
    ID = 0
    
    def __init__(self, config: LearnerConfig): #Use a config dataclass
        self.ID = Learner.ID + 1
        Learner.ID +=1
        
        self.ltm = LongTermMemory(config)
        self.wm = WorkingMemory(self,config) # alpha, 
        self.history = LearningHistory()
        self.chaining = config.chaining
        
        
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
            if not self.chaining:
                s1, s2_index = self.wm.respond(stimuli_stream, s1, s2_index)
            else:
                s1, s2_index = self.wm.respond_with_chaining2(stimuli_stream, s1, s2_index)

        self.final_index = s2_index
        
        
        