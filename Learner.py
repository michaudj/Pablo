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
from TypeNew import Type, TChunk, VChunk

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
    bad_type_threshold: float = -3
    good_type_threshold: float = 2.
    tau: float = 2.0 # softmax parameter of TypeAssigner
    type_on: bool = False
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
            
    def decay_chunk_type_values(self):
        multiplier = 0.999
        for c in self.chunk_type_associations:
            for t in self.chunk_type_associations[c]:
                if self.chunk_type_associations[c][t] > 0:
                    self.chunk_type_associations[c][t] *= multiplier
                
                
    def clean_chunk_type_associations(self):
        elements_to_clean = []
        for c in self.chunk_type_associations:
            for t in self.chunk_type_associations[c]:
                if np.abs(self.chunk_type_associations[c][t]) < 0.5:
                    elements_to_clean.append((c,t))
                    
        for (c,t) in elements_to_clean:
            self.chunk_type_associations[c].pop(t)
        
    def display_typings_of_elements(self):
        for c,d in self.chunk_type_associations.items():
            if len(c) ==1:
                print(f'The type of {c} are {d}')
                
    def display_typings_of_elements2(self):
        for c,d in self.chunk_type_associations.items():
            if len(c) ==1:
                for t,v in d.items():
                    if v > 0:
                        print(f'The type of {c} are {t} with value {v}')
            



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

    def record(self, success_value: int, length: int, verbose=True):
        self.success.append(success_value)
        self.sent_len.append(length)
        if verbose:
            print('----------------------')
            print("Trial", len(self.success))


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
        plt.ylim((0,1))
        plt.title('Learning Progress')
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        return ma
        

    
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
        self.ts1 = TChunk(Type.EMPTY)
        self.ts2 = TChunk(Type.EMPTY)
        self.typing_used = False
        self.border_before = True
        self.border_within = False
        self.border_type = config.border
        self.beta = config.beta
        self.pos = config.positive_reinforcement
        self.neg = config.negative_reinforcement
    
    def get_responses(self):
        responses = []
        for e in self.events:
            responses.append(e[1])
        return responses
    
    def respond(self,stimuli_stream,s1,s2_index,reinforcement = True):
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.read_stimuli(s2_index))
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
        
        pair = ChunkPair((s1,s2))

        # Update memory
        self.learner.ltm.update_repertoire(pair)

        response = self.choose_behaviour(pair)

        self.events.append((pair,response))
        # print(self.get_responses())
        
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
    
    def respond_with_type(self,stimuli_stream,s1,s2_index,reinforcement = True,verbose=True):
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.read_stimuli(s2_index))
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
        
        pair = ChunkPair((s1,s2))
        
        if verbose:
            print("pair",pair)
            print("type 1 before assign: " ,self.learner.wm.ts1)
        self.learner.ltm.update_repertoire(pair)
        
        self.type_assigner.assign_type2(pair) # self.ts1 is a TChunk, while ts2 is a TChunk 
        if verbose:
            print("types after assign, 1: " ,self.learner.wm.ts1, "2: " , self.learner.wm.ts2)
        
        # Check if the structure of ts1 and ts2 are useful as a support for decisions
        # If so, use them in the decision making process
        # Otherwise, fallback on the decision making process without types
        if not self.ts1.is_consistent():  
            self.typing_used = False
            response = self.choose_behaviour(pair)
        else: 
            self.typing_used = True
            # I need to fix the following function! Needs to gather support for decisions...
            response = self.choose_behaviour_with_types(pair) # Set also whether self.typing_used is True or False

        self.events.append((pair,response))
        #print(self.get_responses())
        
        if verbose:
            print("ts1",self.ts1)
            print("ts1 consistency =",self.ts1.is_consistent())
           # print(self.get_responses())
            print("ts2",self.ts2)
            if response == 0:
                print('border')
            else:
                print('chunk')
        
        
        
        
        if response == 0: # boundary placement
            self.learner.n_reinf += 1
            self.learner.ltm.decay_chunk_type_values()
            if self.learner.n_reinf % 100 == 0:
                self.learner.ltm.clean_chunk_type_associations()
            sent_length = stimuli_stream.length_current_sent(s2_index - 1)
            
            if self.is_border_correct(stimuli_stream,s2_index):
                reward = self.pos
                self.learner.history.record(1,sent_length,verbose=True)
                if reinforcement:
                    self.reinforcer.reinforce2(self.events,reward)
                    if not self.typing_used:
                       self.type_assigner.type_sentence(pair.s1)
                       self.reinforcer.reinforce_types(self.typing_events,reward)
                    elif self.ts1.reduce() != Type.SENTENCE:
                        self.typing_events = self.extract_typing_events(s1,self.ts1)
                        self.reinforcer.reinforce_types(self.typing_events,self.neg)
                    else:
                        self.typing_events = self.extract_typing_events(s1,self.ts1)
                        self.reinforcer.reinforce_types(self.typing_events,reward)
            else:
                reward = self.neg
                self.learner.history.record(0,sent_length,verbose=True)
                if reinforcement:
                    self.reinforcer.reinforce2(self.events,reward)
                    if self.typing_used and self.ts1.reduce() == Type.SENTENCE:
                       self.typing_events = self.extract_typing_events(s1,self.ts1)
                       self.reinforcer.reinforce_types(self.typing_events,reward)
            
            new_s1, s2_index = self.get_new_s1(stimuli_stream,s2_index, s2)
            
            self.events = []
            self.typing_events = {}
            self.typing_used = False
               
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
        self.ts1 = self.ts1.chunk_at_depth(self.ts2,depth=pair.s1.depth+1-response)
        s2_index+=1 
        return new_s1, s2_index
    
    def get_new_s1(self,stimuli_stream,s2_index,s2):
        if self.border_type == 'next':
            new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
            new_s1 = SChunk(new_s1)
            self.ts1 = TChunk(Type.EMPTY)
            self.ts2 = TChunk(Type.EMPTY)
        else:
            self.border_before = stimuli_stream.border_before[s2_index]
            new_s1,s2_index = s2, s2_index + 1
            self.ts1 = self.ts2
            self.ts2 = TChunk(Type.EMPTY)

        self.border_within = False
        return new_s1, s2_index
        
    def extract_typing_events(self,s1,ts1,mapping=None):
        if mapping is None:
            mapping = {}
            
        schunk = s1
        tchunk = ts1
    
        # Base case: if it's a leaf node (not a list), just map it
        if not isinstance(schunk.structure, list):
            mapping[schunk] = tchunk.structure
            return mapping
    
        # Map the current (composite) chunk
        mapping[schunk] = tchunk.reduce()
    
        # Recurse on left and right subchunks
        s_left = schunk.get_left()
        s_right = schunk.get_right()
        t_left = tchunk.get_left()
        t_right = tchunk.get_right()
    
        self.extract_typing_events(s_left, t_left, mapping)
        self.extract_typing_events(s_right, t_right, mapping)
    
        return mapping
        # Use the structure of self.ts1 to extract the typing events
        
    
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
    
    def get_z_values_type(self,couple):
        list_right_types = self.ts1.right_types()
        t2= self.ts2.structure 
        
        list_types = self.ts1.remove_structure2()
        list_chunk = couple.s1.flatten_structure()
        list_values = []
        for c,t in zip(list_chunk,list_types):
            self.learner.ltm.update_chunk_type_associations(SChunk(c), t)
            list_values.append(self.learner.ltm.chunk_type_associations[SChunk(c)][t])

        responses = self.get_responses()
            
        value_chunk = VChunk.from_list_and_responses(list_values, responses)
        right_values = value_chunk.right_values()
        #print('-----------')
        #print(right_values)
        # Here, I need to get the elements of s1 and s2
        # Construct the VChunk associated, using the chunk_type_associations dictionary
        # Check whether the action is supported and if so, use the correct value to update z
        
        z = np.zeros(len(list_right_types)+1)
        #print(len(z))
        
        reduced_type = list_right_types[-1]
        if reduced_type == Type.SENTENCE and not reduced_type.is_compatible(t2):
            # print('Support for border, s1 well-typed')
            z[0] = right_values[-1]
        
        for i in range(len(list_right_types)):
            t1 = list_right_types[i]
            if t1.is_compatible(t2):
                #pass
                # print('Support for chunking')
                z[i+1]=right_values[i]
                break
        return z

    def get_right_values(self,couple):
        list_right_types = self.ts1.right_types()
        t2= self.ts2.structure 
        
        list_types = self.ts1.remove_structure2()
        list_chunk = couple.s1.flatten_structure()
        list_values = []
        for c,t in zip(list_chunk,list_types):
            self.learner.ltm.update_chunk_type_associations(SChunk(c), t)
            list_values.append(self.learner.ltm.chunk_type_associations[SChunk(c)][t])

        responses = self.get_responses()
            
        value_chunk = VChunk.from_list_and_responses(list_values, responses)
        right_values = value_chunk.right_values()
        return right_values

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

    def choose_behaviour_with_types(self, couple):
        
        b_range = len(self.learner.ltm.behaviour_repertoire[couple])
        z = self.Q_tilde(couple,b_range)
        z_type = self.get_z_values_type(couple)
        # combine z and z_type with some rules
        z = (z + z_type)/2
        weights = np.exp(self.beta * z)
        options = [i for i in range(b_range)]
        response = random.choices(options,weights/np.sum(weights))
        return response[0]
            
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
    
    def assign_type2(self, pair: ChunkPair):
        # Do type assignment taking into account the values associated to chunk and types
        left_candidates = self.extract_good_starting_types(pair.s1)
        right_candidates = self.extract_good_types(pair.s2)
        
        # Softmax a type for s2
        if right_candidates:
            choice = softmax_choice(right_candidates,tau = self.tau)
            self.learner.wm.ts2 = TChunk(choice)
         
        # s1 simple: assign type if not already done
        if not isinstance(self.learner.wm.ts1.structure,list): # s1 simple
            if self.learner.wm.ts1.has_empty_elements(): # not typed
                if left_candidates: # types available
                    choice = softmax_choice(left_candidates,tau = self.tau)
                    self.learner.wm.ts1 = TChunk(choice)
            else: # already typed, need to check whether it works at the start of a sentence
                if not self.learner.wm.ts1.structure.is_start():
                    if left_candidates:
                        choice = softmax_choice(left_candidates,tau = self.tau)
                        self.learner.wm.ts1 = TChunk(choice)
                    else:
                        self.learner.wm.ts1 = TChunk(Type.EMPTY)
        print("types after first assignment 1: ", self.learner.wm.ts1, "2: ",self.learner.wm.ts2) #ta bort sen
                   
        self.fill_empty_types(pair)
        print("types after fill empty types 1: ", self.learner.wm.ts1, "2: ",self.learner.wm.ts2) #ta bort sen
        
        self.correct_typings2(pair)
        print("types after correct typings 1: ", self.learner.wm.ts1, "2: ",self.learner.wm.ts2)   #ta bort sen 

    
    def assign_type(self, pair: ChunkPair):
        # Do type assignment taking into account the values associated to chunk and types
        left_candidates = self.extract_good_starting_types(pair.s1)
        right_candidates = self.extract_good_types(pair.s2)

        if self.learner.wm.ts1.has_empty_elements():
            # s1 not well-typed
            if isinstance(self.learner.wm.ts1.structure,list):
                #print('bad TCHUNK... Assign ts2 to its best candidate (in case it is used as the beginning of the next sentence)')
                if right_candidates:
                    choice = softmax_choice(right_candidates,tau = self.tau)
                    self.learner.wm.ts2 = TChunk(choice)
            else: # s1 is not complex and untyped...
                #print('Here I should try to assign t1 and t2 jointly') But I do that independently...
                if right_candidates:
                    # Here I need to check for consistency
                    choice = softmax_choice(right_candidates,tau = self.tau)
                    self.learner.wm.ts2 = TChunk(choice)
                if left_candidates:
                    # Here I need to check for consistency
                    choice = softmax_choice(left_candidates,tau = self.tau)
                    self.learner.wm.ts1 = TChunk(choice)

        else:
            #print('s1 typed')
            if right_candidates:
                # Here I need to check for consistency. Here, I softmax a type for s2
                choice = softmax_choice(right_candidates,tau = self.tau)
                self.learner.wm.ts2 = TChunk(choice)
            if not isinstance(self.learner.wm.ts1.structure,list): # s1 is simple
                if not self.learner.wm.ts1.structure.is_start(): # If the type is not a starting type, retype
                    # Here I need to check for consistency
                    if left_candidates:
                        choice = softmax_choice(left_candidates,tau = self.tau)
                        self.learner.wm.ts1 = TChunk(choice)
                        
        # This part should try to find a valid starting type
        if not isinstance(self.learner.wm.ts1.structure, list):
            if not self.learner.wm.ts1.structure.is_start():
                if left_candidates:
                    choice = softmax_choice(left_candidates,tau = self.tau)
                    self.learner.wm.ts1 = TChunk(choice)
                else:
                    self.learner.wm.ts1 = TChunk(Type.EMPTY)
        
        self.fill_empty_types(pair)
        
        self.correct_typings2(pair)
        
        # Check if s1 is typed (no EMPTY types in ts1)
        # if it is
        # - get candidates type for s2 and bad types for s2
        # - Try to find a ts2 compatible with ts1
        # - if found assign it to ts2
        # - if not found, check if ts1 is expecting something after
        # - if so fulfill expectation if proposed type is not bad
        # - otherwise type ts2 as empty (failure to type)
        # if s1 is not typed:
            # collect candidates types for both s1 and s2.
            # Try to find find compatible types and assign the corresponding types to ts1 and ts2
            # Special cases, only s1 has good types or only s2 have good types. 
            # In that case, chose randomly a type for s1 or s2, if it expects something in the other position, fullfil expectation otherwise failure to type
        #self.learner.wm.ts2 = TChunk(Type.EMPTY)
        
    def fill_empty_types(self, pair: ChunkPair):
        
        if not isinstance(self.learner.wm.ts1.structure, list): # ts1 is not complex
            bad_t1 = self.extract_bad_types(pair.s1)
            if self.learner.wm.ts1.has_empty_elements() and not self.learner.wm.ts2.has_empty_elements():
                if self.learner.wm.ts2.structure.is_expecting_before():  
                    # I need to check whether new_ts1 is bad for s1, otherwise, I will assign a bad type!
                    new_ts1 = Type(self.learner.wm.ts2.structure.left_type())
                    if new_ts1 not in bad_t1:
                        self.learner.wm.ts1 = TChunk(new_ts1)
                # elif not self.learner.wm.ts2.structure.is_expecting_before and not self.learner.wm.ts2.structure.is_sentence():
                #     t2_head = Type(self.learner.wm.ts2.structure.left_type()) # what if t2_head is a sentence?????
                #     # while t2_head.is_expecting_after:
                #     #     t2_head_l = Type(t2_head.left_type())
                #     #     t2_head = t2_head_l + t2_head
                #     new_ts1 = Type(Type.SENTENCE.FORMULA + 'o' + t2_head.formula)
                #     # print("new ts1 pre split", new_ts1, "t2 head", t2_head)
                #     # [new_ts1,_] = new_ts1.split(pu=0,prim=t2_head)
                #     # print("new ts1 post split", new_ts1)
                #     if new_ts1 not in bad_t1 and not t2_head.is_sentence():
                #         self.learner.wm.ts1 = TChunk(new_ts1) #Jerome, jag fattar inte riktigt när man ska tchunka och varför. vi har väl inte o-t-chunkat när vi hämtat den från wm?

            elif not self.learner.wm.ts1.has_empty_elements() and self.learner.wm.ts2.has_empty_elements(): 
                bad_t2 = self.extract_bad_types(pair.s2)
                if self.learner.wm.ts1.structure.is_expecting_after(): 
                    # Same here, new_ts2 should not be bad for s2! Otherwise, I will assign a bad type!                    
                    new_ts2 = Type(self.learner.wm.ts1.structure.right_type())
                    if new_ts2 not in bad_t2:
                        self.learner.wm.ts2 = TChunk(new_ts2)
                # elif not self.learner.wm.ts1.structure.is_sentence() and not self.learner.wm.ts1.structure.is_expecting_before():
                #     new_ts2 = Type(self.learner.wm.ts1.structure.formula+'u'+Type.SENTENCE.formula)
                #     # print("new ts2 pre split", new_ts2, "t1", self.learner.wm.ts1.structure)
                #     # [_,new_ts2] = new_ts2.split(pu=1,prim=self.learner.wm.ts1.structure)
                #     # print("new ts2 post split", new_ts2)
                #     if new_ts2 not in bad_t2:
                #         self.learner.wm.ts2 = TChunk(new_ts2)
                    
                    
        elif self.learner.wm.ts1.is_consistent():
            reduced_type = self.learner.wm.ts1.reduce()
            if reduced_type.is_expecting_before():
                print('bad s1')
                sys.exit('bad complex s1')
            if self.learner.wm.ts2.has_empty_elements():
                if reduced_type.is_expecting_after(): 
                    print('This case applies')
                    new_ts2 = Type(reduced_type.right_type())
                    # Here, I also need to check that new_ts2 is not bad for s2!
                    bad_t2 = self.extract_bad_types(pair.s2)
                    if new_ts2 not in bad_t2:
                        self.learner.wm.ts2 = TChunk(new_ts2)
                # elif not reduced_type.is_expecting_after() and not reduced_type.is_sentence() : 
                #     new_ts2 = Type(reduced_type.formula + 'u' + Type.SENTENCE.formula)
                #     # print("new ts2 pre split", new_ts2, "reduced t1", reduced_type)
                #     # [_,new_ts2] = new_ts2.split(pu=1,prim=reduced_type)
                #     # print("new ts2 post split", new_ts2)
                #     if new_ts2 not in bad_t2:
                #         self.learner.wm.ts2 = TChunk(new_ts2)
                    
        elif not self.learner.wm.ts2.has_empty_elements() and self.learner.wm.ts1.has_empty_elements():
            if self.learner.wm.ts2.structure.is_expecting_before(): 
                #print('Retyping complex s1 when s2 expects before and s1 badly typed')
                new_ts1 = Type(self.learner.wm.ts2.structure.left_type())
                #print(f"Expected type is {new_ts1}")
                # s1 complex and not well-typed and s2 expecting before, should retype the complex s1...
                leaf_types = self.infer_leaf_types(pair.s1, new_ts1)
                #print(f"The list of types at the leaves are: {leaf_types}")
                responses = self.learner.wm.get_responses()
                self.learner.wm.ts1 = TChunk.from_list_and_responses(leaf_types, responses)
            # elif not self.learner.wm.ts2.structure.is_expecting_before():
            #     #s1 should expect head of s2 and reduce to sentence with it
            #     print ("fill compound empty type 1 to expect t2 head") #ta bort sen
            #     t2_head = Type(self.learner.wm.ts2.structure.left_type())
            #     #while t2_head.is_expecting_after:
            #     #    t2_head_r = Type(t2_head.right_type())
            #     #    t2_head = t2_head + t2_head_r
            #     new_ts1 = Type(Type.SENTENCE.formula + 'o' + t2_head.formula)    
            #     # [new_ts1,_] = new_ts1.split(pu=0,prim=t2_head)
                 
            #     if not t2_head.is_sentence():
            #         leaf_types = self.infer_leaf_types(pair.s1, new_ts1)
            #         responses = self.learner.wm.get_responses()
            #         self.learner.wm.ts1 = TChunk.from_list_and_responses(leaf_types, responses)
            #         print("filled compound type 1 ",new_ts1 )
                
                
        #elif ts1 is not consistent but there are expectations lower down the structure?
        #else ts1 inconsistent and no expectations, if ts2 is expecting before, it should type s1 using a similar procedure than when a sentence is typed for the first time...
        # This is one way to change the head of type...


            
            # if ts1 not a list and empty and ts2 non empty and expecting before
            # assign expectation to ts1
            # elif ts1 is consistent and expecting after and ts2 empty
            # assign expectation to ts2


    def correct_typings(self, pair: ChunkPair):
        if not self.learner.wm.ts1.has_empty_elements() and not self.learner.wm.ts2.has_empty_elements():
            if not isinstance(self.learner.wm.ts1.structure,list) and not isinstance(self.learner.wm.ts2.structure, list):
                # print('Both non complex')
                t1 = self.learner.wm.ts1.structure
                t2 = self.learner.wm.ts2.structure
                # Should I use the compatibility check from the type class here?
                if t1.is_expecting_after() and not t2.is_expecting_before():
                    # print('t1 expectations')
                    # Check compatibility and correct if needed
                    # print(f't1 {t1} is expecting after {Type(t1.right_type())} and t2 is {t2}')
                    t1_r = Type(t1.right_type())
                    if t1_r == t2:
                        pass
                        # print('Good match')
                    else:
                        # print('Bad match')
                        # Check if expectation is a bad match for t2
                        bad_t2 = self.extract_bad_types(pair.s2)
                        good_t2 = self.extract_good_types(pair.s2)
                        if t1_r in bad_t2: # or t1_r has not been used for that element
                            # print('Should retype expectation')
                            new_t1 = t1 + t1_r
                            [new_t1,t2] = new_t1.split(pu=0,prim=t2)
                            self.learner.wm.ts1 = TChunk(new_t1)
                            #self.learner.wm.ts2 = TChunk(t2)
                        else:
                            self.learner.wm.ts2 = TChunk(t1_r)
                        # retype here
                elif not t1.is_expecting_after() and t2.is_expecting_before():
                    # print('t2 expectations')
                    # print(f't2 {t2} is expecting before {Type(t2.left_type())} and t1 is {t1}')
                    # Check compatibility and correct if needed
                    t2_l = Type(t2.left_type())
                    if t2_l == t1:
                        pass
                        #print('Good match')
                    else:
                        # print('Bad match')
                        # Check if expectation is a bad type for t1
                        bad_t1 = self.extract_bad_types(pair.s1)
                        good_t1 = self.extract_good_types(pair.s1)
                        if t2_l in bad_t1:
                            # print('Should retype expectation')
                            new_t2 = t2_l+t2
                            [t1,new_t2] = new_t2.split(pu=1,prim=t1)
                            self.learner.wm.ts2 = TChunk(new_t2)
                            #self.learner.wm.ts1 = TChunk(t1)
                        else:
                            self.learner.wm.ts1 = TChunk(t2_l)
                            # print(t1)
                            # print(new_t2)
                            # add t2 to is left type and split it using t1
                        # retype here
                    pass
                elif t2.is_expecting_before() and t1.is_expecting_after():
                    # Incompatible types! Try to find a compatible pairing
                    print(f'Incompatible typing at step {self.learner.n_reinf}')
                    # retype here
                
            elif self.learner.wm.ts1.is_consistent():
                #print(self.learner.wm.get_responses())
                # print('ts1 complex')
                reduced_type = self.learner.wm.ts1.reduce()
                t2 = self.learner.wm.ts2.structure
                if reduced_type.is_expecting_after() and not t2.is_expecting_before():
                    rt_r = Type(reduced_type.right_type())
                    bad_t2 = self.extract_bad_types(pair.s2)
                    good_t2 = self.extract_good_types(pair.s2)
                    if rt_r in bad_t2: # or t1_r has not been used for that element
                        # print('Should retype expectation')
                        
                        # print(f'ts1 is {self.learner.wm.ts1} and ts2 is {self.learner.wm.ts2}')
                        if t2.is_primitive():# and len(pair.s1) <=2:
                            # print(f't2 {t2} is a primitive type')
                            new_reduced = Type(reduced_type.formula + 'o' + t2.formula)
                            self.learner.wm.ts1 = self.learner.wm.ts1.retype_root(new_reduced,self.learner.wm.get_responses())
                            # self.learner.wm.ts1 = self.learner.wm.ts1.retype_expectation(t2,self.learner.wm.get_responses())
                            # print(f'ts1 consistent after changing expectation: {self.learner.wm.ts1.is_consistent()}')

                        # print('retyping')
                        # print(f'ts1 is {self.learner.wm.ts1} and ts2 is {self.learner.wm.ts2}')
                        
                        # new_t1 = t1 + t1_r
                        # [new_t1,t2] = new_t1.split(pu=0,prim=t2)
                        # self.learner.wm.ts1 = TChunk(new_t1)
                        # self.learner.wm.ts2 = TChunk(t2)
                    else:
                        self.learner.wm.ts2 = TChunk(rt_r)
                    #self.learner.wm.ts2 = TChunk(new_ts2)
                # ts1 complex: check if it reduces to something that expect something after. If ts2 expects something before retype ts2
                
    def correct_typings2(self, pair: ChunkPair):
       if not self.learner.wm.ts1.has_empty_elements() and not self.learner.wm.ts2.has_empty_elements():
           if not isinstance(self.learner.wm.ts1.structure,list) and not isinstance(self.learner.wm.ts2.structure, list):
               # print('Both non complex')
               t1 = self.learner.wm.ts1.structure
               t2 = self.learner.wm.ts2.structure
               if t1.is_expecting_after() and not t2.is_expecting_before():
                   # print('t1 expectations')
                   # Check compatibility and correct if needed
                   # print(f't1 {t1} is expecting after {Type(t1.right_type())} and t2 is {t2}')
                   t1_r = Type(t1.right_type())
                   if not t1.is_compatible(t2):
                       # print('correct typings here (case 1)')
                       # print(self.learner.wm.ts1)
                       # print(self.learner.wm.ts2)
                       self.learner.ltm.update_chunk_type_associations(pair.s1, t1)
                       self.learner.ltm.update_chunk_type_associations(pair.s2, t2)
                       value_ts1 = self.learner.ltm.chunk_type_associations[pair.s1][t1]
                       value_ts2 = self.learner.ltm.chunk_type_associations[pair.s2][t2]
                       
                       candidates = {'s1': value_ts1,'s2': value_ts2}
                       dominant_side = softmax_choice(candidates, tau=self.tau)
                       
                       bad_t2 = self.extract_bad_types(pair.s2)
                       good_t2 = self.extract_good_types(pair.s2)
                       
                       if t1_r in bad_t2 or dominant_side == "s2": # or t1_r has not been used for that element
                           # print('Should retype expectation')
                           new_t1 = t1 + t1_r
                           [new_t1,t2] = new_t1.split(pu=0,prim=Type(t2.left_type()))
                           self.learner.wm.ts1 = TChunk(new_t1)
                           # self.learner.wm.ts2 = TChunk(t2)
                       else:
                           self.learner.wm.ts2 = TChunk(t1_r)
                           
                       # print('new types')
                       # print(self.learner.wm.ts1)
                       # print(self.learner.wm.ts2)
                   
                       
               elif not t1.is_expecting_after() and t2.is_expecting_before():
                   t2_expect_too_much = False
                   # print('t2 expectations')
                   # print(f't2 {t2} is expecting before {Type(t2.left_type())} and t1 is {t1}')
                   # Check compatibility and correct if needed
                   t2_l = Type(t2.left_type())
                   #if t1.is_compatible(t2):
                   reduced_t2 = t2_l+t2
                   if reduced_t2.is_expecting_before():
                       t2_expect_too_much =True
                       # print("t2 too much")
                   
                   if not t1.is_compatible(t2) or t2_expect_too_much:
                       # print("not compatible or t2 too much")
                       # print("t1", t1, "t2", t2)
                       # print('correct typings here (case 2)')
                       # print(self.learner.wm.ts1)
                       # print(self.learner.wm.ts2)
                       self.learner.ltm.update_chunk_type_associations(pair.s1, t1)
                       self.learner.ltm.update_chunk_type_associations(pair.s2, t2)
                       value_ts1 = self.learner.ltm.chunk_type_associations[pair.s1][t1]
                       value_ts2 = self.learner.ltm.chunk_type_associations[pair.s2][t2]
                       
                       candidates = {'s1': value_ts1,'s2': value_ts2}
                       dominant_side = softmax_choice(candidates, tau=self.tau)
                       
                       bad_t1 = self.extract_bad_types(pair.s1)
                       good_t1 = self.extract_good_types(pair.s1)
                       if t2_l in bad_t1 or dominant_side == "s1" or t2_expect_too_much:
                           # print('Should retype expectation')
                           
                           new_t2 = t2_l+t2  #takes away backwards expectation via reduction
                           # print("first reduction", t2_l, "+" , t2, "=" , new_t2)
                           #in the case of t1 being primitive and dominant, and t2 expecting too much backwards, t2 only keeps its head and is modified to expect t1 backwards. fixed below.
                           while new_t2.is_expecting_before():
                               t2_l = Type(new_t2.left_type())
                               # print("next reduction", t2_l, "+" , new_t2)
                               new_t2 = t2_l + new_t2
                           [t1,new_t2] = new_t2.split(pu=1,prim=t1) #adds new expectation to t2 via split
                           self.learner.wm.ts2 = TChunk(new_t2)
                           #self.learner.wm.ts1 = TChunk(t1)
                           
                       else: #means none of conditions t2_l in bad t1 or dominant side is s1, i.e. dominant side is s2 do it should assign its expectation to t1. done already below.
                           self.learner.wm.ts1 = TChunk(t2_l)
                   # else:
                   #     pass
                   #     # if t1+t2 check if expecting before retype t2
                        
                
                           
                           
                       # print('new types')
                       # print(self.learner.wm.ts1)
                       # print(self.learner.wm.ts2)
                   
               elif t2.is_expecting_before() and t1.is_expecting_after():
                   # Incompatible types! Try to find a compatible pairing
                   #print(f'Incompatible typing at step {self.learner.n_reinf}')
                   # print("both sides expecting")
                   # retype here
                   self.learner.ltm.update_chunk_type_associations(pair.s1, t1)
                   self.learner.ltm.update_chunk_type_associations(pair.s2, t2)
                   value_ts1 = self.learner.ltm.chunk_type_associations[pair.s1][t1]
                   value_ts2 = self.learner.ltm.chunk_type_associations[pair.s2][t2]
                   
                   candidates = {'s1': value_ts1,'s2': value_ts2}
                   dominant_side = softmax_choice(candidates, tau=self.tau)
                   
                   bad_t1 = self.extract_bad_types(pair.s1)
                   good_t1 = self.extract_good_types(pair.s1)
                   bad_t2 = self.extract_bad_types(pair.s2)
                   good_t2 = self.extract_good_types(pair.s2)
                   
                   t1_r= Type(t1.right_type())
                   t2_l= Type(t2.left_type())
                   
                   if t1_r in bad_t2:
                       dominant_side = "s2"
                   if t2_l in bad_t1:
                       dominant_side = "s1"
                   reduced_t2 = t2_l+t2
                   if reduced_t2.is_expecting_before():
                       dominant_side = "s1"
                       
                   if dominant_side == "s1":
                       t2=t1_r
                       self.learner.wm.ts1 = TChunk(t1)
                       self.learner.wm.ts2 = TChunk(t2)
                   elif dominant_side == "s2":
                       t1=t2_l
                       self.learner.wm.ts1 = TChunk(t1)
                       self.learner.wm.ts2 = TChunk(t2)
                           
           elif self.learner.wm.ts1.is_consistent():
               #print(self.learner.wm.get_responses())
               # I NEED TO CHECK WHETHER THE REDUCED TYPE IS A GOOD STARTING TYPE, OTHERWISE IT SHOULD BE RETYPED! (A: no, now solved in previous round)
               
               reduced_type = self.learner.wm.ts1.reduce() 
               # print("reduced type1", reduced_type)
               
               if reduced_type.is_expecting_before(): 
                   print("error: inherited type1 is expecting before:",reduced_type)
                   sys.exit("Too much expectations")

               t2 = self.learner.wm.ts2.structure
               
               if reduced_type.is_expecting_after() and not t2.is_expecting_before():
                   rt_r = Type(reduced_type.right_type())
                   if not reduced_type.is_compatible(t2):                   
                       right_types = self.learner.wm.ts1.right_types()

                       for i,t in enumerate(right_types):
                           if t.is_expecting_after():
                               valueindex = i
                               
 
                       right_values = self.learner.wm.get_right_values(pair)
                       competing_t1_value = right_values[valueindex] 

                       self.learner.ltm.update_chunk_type_associations(pair.s2, t2)
                       value_ts2 = self.learner.ltm.chunk_type_associations[pair.s2][t2]

                       candidates = {'s1': competing_t1_value,'s2': value_ts2}
                       dominant_side = softmax_choice(candidates, tau=self.tau)
                       bad_t2 = self.extract_bad_types(pair.s2)
                       good_t2 = self.extract_good_types(pair.s2)
                       if rt_r in bad_t2 or dominant_side == 's2': # or t1_r has not been used for that element

                           # if t2.is_primitive():# and len(pair.s1) <=2:
                               # print(f'ts1: {self.learner.wm.ts1}')
                               # print(f'ts2: {self.learner.wm.ts2}')
                               new_t1 = reduced_type + rt_r
                               # print("new t1", new_t1, "prim for split ", Type(t2.left_type()))
                               [new_t1,_] = new_t1.split(pu=0,prim=Type(t2.left_type()))
                               # print("new new t1 ", new_t1, "new t2 ", t2)
                               # print(f'Retype to new_ts1 {new_t1} and t2 {t2}')
                               self.learner.wm.ts1 = self.learner.wm.ts1.retype_root(new_t1,self.learner.wm.get_responses())
                               #self.learner.wm.ts1 = self.learner.wm.ts1.retype_expectation(t2,self.learner.wm.get_responses())
                               # print("final ts1", self.learner.wm.ts1)
                       else:
                           self.learner.wm.ts2 = TChunk(rt_r)
               elif not reduced_type.is_expecting_after() and t2.is_expecting_before():
                   t2_expect_too_much = False
                   t2_l = Type(t2.left_type())
                   reduced_t2 = t2_l+t2
                   if reduced_t2.is_expecting_before():
                       t2_expect_too_much =True
                       # print("t2 too much")
                   if not reduced_type.is_compatible(t2) or t2_expect_too_much: # add condition for sentence
                       # print('t2 is expecting before, so t1 should be retyped or the expectation of t2 should be changed')
                       # print(f't2 is {t2}')
                       # print(f't1 is {reduced_type}')
                       # print(f'ts1 is {self.learner.wm.ts1}')
                       right_types = self.learner.wm.ts1.right_types()
                       # print(right_types)

                               
 
                       right_values = self.learner.wm.get_right_values(pair)
                       # print(right_values)
                       competing_t1_value = right_values[-1] 

                       self.learner.ltm.update_chunk_type_associations(pair.s2, t2)
                       value_ts2 = self.learner.ltm.chunk_type_associations[pair.s2][t2]

                       candidates = {'s1': competing_t1_value,'s2': value_ts2}
                       dominant_side = softmax_choice(candidates, tau=self.tau)
                       # print("dominant side", dominant_side)
                       if dominant_side == "s1" or t2_expect_too_much:
                           # print('Should retype expectation')
                           
                           new_t2 = t2_l+t2  #takes away backwards expectation via reduction
                           # print("first reduction", t2_l, "+" , t2, "=" , new_t2)
                           #here we can check if new t2 is a primitive, if not, we reduce again, to make sure it will only take one argument backwards to not end up wit an inherited s1 with backwards expectation
                           while new_t2.is_expecting_before():
                               t2_l = Type(new_t2.left_type())
                               # print("next reduction", t2_l, "+" , new_t2)
                               new_t2 = t2_l + new_t2                                                                                      
                           [_,new_t2] = new_t2.split(pu=1,prim=reduced_type)
                           self.learner.wm.ts2 = TChunk(new_t2)
                           # self.learner.wm.ts1 = TChunk(t1)
                       else:
                           self.learner.wm.ts1 = self.learner.wm.ts1.retype_root(t2_l,self.learner.wm.get_responses())
                                           
               elif reduced_type.is_expecting_after() and t2.is_expecting_before():
                   print("t1 complex, both sides expecting")
                   # retype here
                   #self.learner.ltm.update_chunk_type_associations(pair.s1, t1)
                   self.learner.ltm.update_chunk_type_associations(pair.s2, t2)
                   #value_ts1 = self.learner.ltm.chunk_type_associations[pair.s1][t1]
                   value_ts2 = self.learner.ltm.chunk_type_associations[pair.s2][t2]
                  
                   right_types = self.learner.wm.ts1.right_types()
    
                   for i,t in enumerate(right_types):
                       if t.is_expecting_after():
                           valueindex = i                          
                   right_values = self.learner.wm.get_right_values(pair)
                   competing_t1_value = right_values[valueindex] 
                   candidates = {'s1': competing_t1_value,'s2': value_ts2}
                   dominant_side = softmax_choice(candidates, tau=self.tau)
                   
                    #bad_t1 = self.extract_bad_types(pair.s1)
                    #good_t1 = self.extract_good_types(pair.s1)
                   bad_t2 = self.extract_bad_types(pair.s2)
                   good_t2 = self.extract_good_types(pair.s2)
                   
                   t1_r= Type(reduced_type.right_type())
                   t2_l= Type(t2.left_type())
                   
                   if t1_r in bad_t2:
                       dominant_side = "s2"
                    #if t2_l in bad_t1:
                        #dominant_side = "s1"
                   reduced_t2 = t2_l+t2
                   if reduced_t2.is_expecting_before():
                       dominant_side = "s1"    
                   print("dominant side" , dominant_side)
                   if dominant_side == "s1" and not t1_r in bad_t2:
                        t2=t1_r
                        #self.learner.wm.ts1 = TChunk(reduced_type)
                        self.learner.wm.ts2 = TChunk(t2)
                        print("corrected t2: ",self.learner.wm.ts2)
                   elif dominant_side == "s2": # and not t2_l in bad_t1:
                       #here the expectation of s2 should be assigned as the head of s1. 
                       #s1 is expectring, does not reduce to a primitive, cannot use retype root?
                        leaf_types = self.infer_leaf_types(pair.s1, t2_l)
                        responses = self.learner.wm.get_responses()
                        self.learner.wm.ts1 = TChunk.from_list_and_responses(leaf_types, responses)
                        #self.learner.wm.ts1 = self.learner.wm.ts1.retype_root(t2_l,self.learner.wm.get_responses())
                        #print("input retype root: ",t2_l)
                        print("corrected t1: ",self.learner.wm.ts1)
                        #leaf_types = self.infer_leaf_types(pair.s1, t2_l)
                       #print(f"The list of types at the leaves are: {leaf_types}")
                       #responses = self.learner.wm.get_responses()
                       #print(leaf_types, responses)
                       #self.learner.wm.ts1 = TChunk.from_list_and_responses(leaf_types, responses)#gör original split på s1 med t2l som head
                   
                   
           elif not self.learner.wm.ts1.is_consistent():
                 t2 = self.learner.wm.ts2.structure
                 if t2.is_expecting_before():
                     full_t1 = self.learner.wm.ts1
                     flattened_t1 = full_t1.remove_structure2()  
                     # print("flat list",flattened_t1)
                     max_reduced_t1 = Type.reduce(flattened_t1) 
                     all_primitives = True
                     for typ in max_reduced_t1:
                         if not typ.is_primitive():
                             all_primitives = False
                             break
                     if all_primitives:
                         values_ts1 = []
                         flat_s1 = pair.s1.flatten_structure()
                         for i in range(len(flattened_t1)):
                             self.learner.ltm.update_chunk_type_associations(flat_s1[i], flattened_t1[i])
                             values_ts1.append(self.learner.ltm.chunk_type_associations[flat_s1[i]] [flattened_t1[i]])
                             value_ts1 = sum(values_ts1)/len(values_ts1)
                         self.learner.ltm.update_chunk_type_associations(pair.s2, t2)                   
                         value_ts2 = self.learner.ltm.chunk_type_associations[pair.s2][t2]
                         candidates = {'s1': value_ts1,'s2': value_ts2}
                         dominant_side = softmax_choice(candidates, tau=self.tau)
                         t2_l= Type(t2.left_type())
                         reduced_t2 = t2_l+t2
                         if reduced_t2.is_expecting_before():
                             dominant_side = "s1" #temporary solution, in future we should check if number of s2 backwards expectation matches number of elements in max reduced s1
                        
                         if dominant_side == "s1":
                         #first remove all backwards expectations of t2, then add all on max reduced t1
                             while reduced_t2.is_expecting_before():
                                 reduced_t2_l = Type(reduced_t2.left_type())
                                 reduced_t2 = reduced_t2_l+reduced_t2
                             formula = reduced_t2.formula
                             for typ in max_reduced_t1:
                                 formula = typ.formula + 'u' + formula
                             t2 = Type(formula)
                             self.learner.wm.ts2 = TChunk(t2)
                         elif dominant_side == "s2":
                            
                             # print("s2 is expecting one before and this will be assigned to the inconsistent t1")
                             leaf_types = self.infer_leaf_types(pair.s1, t2_l)
                             #print(f"The list of types at the leaves are: {leaf_types}")
                             responses = self.learner.wm.get_responses()
                             self.learner.wm.ts1 = TChunk.from_list_and_responses(leaf_types, responses)#gör original split på s1 med t2l som head, fråga jorpan it is fill empty types
               

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
        
    def is_new(self,typ:Type, chunk: SChunk):
        if chunk in self.learner.ltm.chunk_type_associations:
            if typ in self.learner.ltm.chunk_type_associations[chunk]:
                return True
            else:
                return False
        else:
            return False
      
    def infer_leaf_types(self, s: SChunk, current_type: Type):
        if not isinstance(s.structure, list):
            # It's a leaf
            return [current_type]
    
        # It's a binary structure: get children
        left_s = s.get_left()
        right_s = s.get_right()
    
        # Decide the left and right types this current_type splits into
        left_type, right_type = self.choose_types(current_type, left_s, right_s)
    
        # Recurse down and collect all leaf types
        return self.infer_leaf_types(left_s, left_type) + self.infer_leaf_types(right_s, right_type)

    
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
        
    def extract_good_starting_types(self, chunk: SChunk):
        def filter_dict_above_threshold(data,threshold):
            result = {k: v for k, v in data.items() if v > threshold and k.is_start()}
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
        self.type_on = config.type_on
        
        
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
                if self.type_on:
                    s1, s2_index = self.wm.respond_with_type(stimuli_stream, s1, s2_index,verbose=True)
                else:
                    s1, s2_index = self.wm.respond(stimuli_stream, s1, s2_index)
            else:
                s1, s2_index = self.wm.respond_with_chaining2(stimuli_stream, s1, s2_index)

        self.final_index = s2_index
        
        
        