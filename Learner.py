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

from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.styles import PatternFill

import matplotlib.pyplot as plt



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
    alpha_v: float = 0.3
    beta: float = 1.
    positive_reinforcement: float = 5.
    negative_reinforcement: float = -10.
    RW: bool = False
    chaining: bool = False
    # parameter for choosing type of learner (RW or not.)

class LongTermMemory():
    
    def __init__(self,config):
        self.initial_value_chunking = config.initial_value_chunking
        self.initial_value_border = config.initial_value_border
        self.behaviour_repertoire = {} # dictionary of where the keys are couples of chunks and the value a list of behavioural values
        self.chunk_values = {}

    
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
        self.events = []
        self.border_before = True
        self.border_within = False
        self.border_type = config.border
        self.beta = config.beta
        self.pos = config.positive_reinforcement
        self.neg = config.negative_reinforcement
        
    def place_border_and_reinforce(self,stimuli_stream,s2,s2_index, reinforcement = True):
        self.learner.n_reinf += 1
        sent_length = stimuli_stream.length_current_sent(s2_index - 1)
        is_border = stimuli_stream.border_before[s2_index]
        if is_border and not self.border_within and self.border_before:
            # Good unit
            if reinforcement:
                self.reinforcer.reinforce2(self.events,self.pos)  
            self.learner.history.record(1,sent_length)
        else:
            # Bad unit
            if reinforcement:
                self.reinforcer.reinforce2(self.events,self.neg) 
            self.learner.history.record(0,sent_length)
            
        new_s1, s2_index = self.get_new_s1(stimuli_stream, s2_index, s2)
        
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
    
    def is_border_correct(self,stimuli_stream,s2_index):
        is_border = stimuli_stream.border_before[s2_index]
        return is_border and not self.border_within and self.border_before
    
    def get_new_s1(self,stimuli_stream,s2_index,s2):
        if self.border_type == 'next':
            new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
            new_s1 = SChunk(new_s1)
        else:
            self.border_before = stimuli_stream.border_before[s2_index]
            new_s1,s2_index = s2, s2_index + 1

        self.border_within = False
        return new_s1, s2_index
        
    
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
            
        # # Reinforce the event and the value of s1.
        # if reinforcement:
        #     self.reinforcer.reinforce2(event,reward)
        #     self.reinforcer.reinforce_value(pair.s1,reward)
            
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
        
        
        