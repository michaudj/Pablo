# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:51:34 2023

@author: jmd01
"""

import json
import copy
import re
import numpy as np
import random
from new_raw_input import ProbabilisticGrammar
#import sys
#sys.setrecursionlimit(1500)

def modify_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = [nested_list[-1],new_value]
    
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def add_weights(b_values1,b_values2):
    a = copy.deepcopy(b_values1)
    b = copy.deepcopy(b_values2)
    l = sorted((a, b), key=len)
    c = l[1].copy()
    c[:len(l[0])] += l[0]
    return c

class Chunk():
    
    cache = {}
    
        # override the __new__ method to check the cache for an existing instance
    def __new__(cls, structure):
        
        # check if the key is in the cache
        if json.dumps(structure) in cls.cache:
            # if the key is in the cache, return the corresponding instance
            return cls.cache[json.dumps(structure)]
        else:
            # if the key is not in the cache, create a new instance
            instance = super().__new__(cls)
            # store the new instance in the cache
            cls.cache[json.dumps(structure)] = instance
            # return the new instance
            return instance
    
    def __init__(self, structure):
        self.structure = structure
        #self.type_dic = {}
        self.depth = self.get_depth()
        
    def __repr__(self):
        return str(self.structure)
    
    def get_right_subchunks(self, depth):
        right_subchunks = []
        nested_list = copy.deepcopy(self.structure)
        for d in range(depth):
            nested_list = nested_list[-1]
            right_subchunks.append(Chunk(nested_list))
        return right_subchunks
    
    def chunk_at_depth(self, other, depth=0):
        nested_list = copy.deepcopy(self.structure)
        
        if depth == 0:
            return Chunk([nested_list,other.structure])
        else:
            modify_element_at_depth(nested_list, depth, other.structure)
            return Chunk(nested_list)
    
    def get_depth(self):
        st = str(self.structure)
        match = re.search("]*$",st)
        return len(match.group(0))
    
    def remove_structure(self):
        if type(self.structure) is str:
            return self.structure
        else:
            return flatten(self.structure)
    
    def get_terminals(self):
        return set(self.remove_structure())



        
        
class Learner():
    
    alpha = 0.2
    beta = 1.
    positive_reinforcement = 5.
    negative_reinforcement = -1.
    initial_value_chunking = -1.
    initial_value_border = 1.
    ID = 0
    
    def __init__(self, n_trials = 14, t = 'flexible', border = 'next'):
        self.ID = Learner.ID + 1
        Learner.ID +=1
        self.type = t
        self.border_type = border # or 'default'
        self.n_trials = n_trials
        self.n_reinf = 0
        self.success = []
        self.sent_len = []
        self.sentences = set()
        # Change this set into a dictionary where the key is a dictionary of types
        # self.chunks = set()
        self.chunk_dict = {} #encodes the long term memory and is a set of chunks (put the types here!)
        self.behaviour_repertoire = {} # dictionary of where the keys are couples of chunks and the value a list of behavioural values
        self.events = [] # encodes the current list of couples ((chunk,chunk), behaviour) to reinforce
        self.border_before = True
        self.border_within = False
        # for grammar extraction
        self.terminals = set()
        self.non_terminals = set()
        self.rules = dict()
        self.weights = dict()
        self.grammar = None
        
    def __repr__(self):
        string = 'Learner ' + str(self.ID)
        
        return string
        
    def add_chunk(self, chunk):
        if chunk not in self.chunk_dict:
            self.chunk_dict[chunk] = {}
            
    def update_repertoire(self,couple):
        if couple not in self.behaviour_repertoire:
            if self.type == 'right':
                self.behaviour_repertoire[couple] = np.array([Learner.initial_value_border] + [Learner.initial_value_chunking] )
            elif self.type == 'flexible':
                self.behaviour_repertoire[couple] = np.array([Learner.initial_value_border] + [Learner.initial_value_chunking for i in range(couple[0].depth+1)])
                #print(Learner.initial_value_chunking)
                #print(self.behaviour_repertoire[couple])
            else:
                print('Learning type not defined')
                
    def get_sub_couples(self, couple):
        sub_pairs = []
        for s in couple[0].get_right_subchunks(couple[0].depth):
            self.add_chunk(s)
            sub_pairs.append((s,couple[1]))
            self.update_repertoire((s,couple[1]))
        return sub_pairs
    

    def learn(self,stimuli_stream):
        #Chunk.clear_cache()
        # initialize stimuli
        s1 = Chunk(stimuli_stream.stimuli[0])
        #self.chunks.add(s1)
        self.add_chunk(s1)
        s2_index = 1
        #for t in range(self.n_trials):
        while self.n_reinf <= self.n_trials:
            s1, s2_index = self.respond(stimuli_stream, s1, s2_index)
    
    def respond(self,stimuli_stream,s1,s2_index):
        # get the s2 stimuli and make it a chunk
        s2 = Chunk(stimuli_stream.stimuli[s2_index])
        # update chunkatory
        self.add_chunk(s2)
        # choose a response for this pair of chunks
        response = self.choose_behaviour((s1,s2))
        # add the behaviour to the events list
        self.events.append(((s1,s2),response))
        
        
        # # Add sub-events to the events list. Maybe not here... Consider moving this in the reinforcement part.
        # if self.type == 'flexible':
        #     for couple in self.get_sub_couples((s1,s2)):
        #         if response < len(self.behaviour_repertoire[couple]):
        #             self.events.append((couple,response))
        #         #print(self.events)
                
        if response == 0: # boundary placement
            # increment the number of reinforcement events by 1
            self.n_reinf += 1 
            # check if border is correctly placed
            is_border = stimuli_stream.border_before[s2_index]
            #
            if is_border and not self.border_within and self.border_before:
                #print('Good Unit')
                # perform positive reinforcement
                # Store sentence (not cognitively plausible but used for grammar extraction)
                self.sentences.add(s1)
                # Perform reinforcement
                self.reinforce(reinforcement = 'positive')                    
                # update the success list
                self.success.append(1)
                # for postprocessing, store length of the sentence
                self.sent_len.append(stimuli_stream.length_current_sent(s2_index - 1))
            else:
                #print('Bad Unit')
                # perform negative reinforcement
                self.reinforce(reinforcement = 'negative')
                # update the success list
                self.success.append(0)
                self.sent_len.append(stimuli_stream.length_current_sent(s2_index))
            # Next beginning of sentence becomes
            
            if self.border_type == 'next':
                new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
                new_s1 = Chunk(new_s1)
            else:
                self.border_before = stimuli_stream.border_before[s2_index]
                new_s1,s2_index = s2, s2_index + 1
                
            #self.chunks.add(new_s1)
            self.add_chunk(new_s1)
            self.border_within = False
            
        else: # some type of chunking occurs
            # Check if there was a border
            if not self.border_within:
                self.border_within = stimuli_stream.border_before[s2_index]
            
            # Perform chunking at correct level
            if self.type == 'right':
                # response is 1 and therefore, chunking occurs
                new_s1 = s1.chunk_at_depth(s2)   
            elif self.type == 'flexible':
                # response >= 1 and chunking or subchunking occurs
                new_s1 = s1.chunk_at_depth(s2,depth=s1.depth+1-response) 
            else:
                print('Wrong type!')

            s2_index+=1
            self.add_chunk(new_s1)
            
        return new_s1, s2_index


    def choose_behaviour(self,couple):
        self.update_repertoire(couple)
        b_range = len(self.behaviour_repertoire[couple])
        options = [i for i in range(b_range)]
        z = copy.deepcopy(self.behaviour_repertoire[couple])
        
        
        if self.type == 'flexible':
            subpairs = self.get_sub_couples(couple)
            norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
            # Accumulate support from subchunks
            for pair in subpairs:
                z = add_weights(z, self.behaviour_repertoire[pair])
            # Take the average
            z /= norm_vec
            weights = np.exp(Learner.beta * z)
        elif self.type == 'right':
            weights = np.exp(Learner.beta * z)
        else:
            print('Unknown learner type')
            
        response = random.choices(options,weights/np.sum(weights))
        #print(z)
        #print(response[0])
        return response[0]  
    
    def reinforce(self, reinforcement = 'positive'):
        # for each events reinforce behaviour associated to chunk
        if reinforcement == 'positive':
            u = Learner.positive_reinforcement
        elif reinforcement == 'negative':
            u = Learner.negative_reinforcement
            
        for couple,r in self.events:
            if self.type == 'right':
                self.behaviour_repertoire[couple][r] += Learner.alpha * (u - self.behaviour_repertoire[couple][r])
            elif self.type == 'flexible':
                #print('reinforcement')
                Q = self.behaviour_repertoire[couple][r]
                subevents = [(couple,r)]
                subpairs = self.get_sub_couples(couple)
                for pair in subpairs:
                    if r < len(self.behaviour_repertoire[pair]):
                        subevents.append((pair,r))
                        Q += self.behaviour_repertoire[pair][r]
                #Q /= len(subevents)
                #print(Q)
                for p,rr in subevents:
                    self.behaviour_repertoire[p][rr] += Learner.alpha * (u - Q)
            else:
                print('ERROR')
        
        # Clear working memory
        self.events = []
        
    def extract_sentences(self, threshold):
        sentences = set()
        for pair in self.behaviour_repertoire:
            if self.behaviour_repertoire[pair][0]> threshold :
                sentences.add(pair[0])
        return sentences
        # this function should loop through self.behaviour_repertoire and look for pairs with high first value and create a set of identified sentences with their structures
        
    def sentences_weights(self):
        sent_weight = dict()
        for pair in self.behaviour_repertoire:
            if pair[0] in self.sentences:
                if pair[0] not in sent_weight:
                    sent_weight[pair] = []
                b_range = len(self.behaviour_repertoire[pair])
                z = copy.deepcopy(self.behaviour_repertoire[pair])
                subpairs = self.get_sub_couples(pair)
                norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
                # Accumulate support from subchunks
                for p in subpairs:
                    z = add_weights(z, self.behaviour_repertoire[p])
                # Take the average
                z /= norm_vec
                sent_weight[pair].append(z[0])
        return sent_weight
    
    def extract_PCFG(self):
        # Extract terminals
        self.terminals = set()
        for s in self.sentences:
            #print(set(s.remove_structure()))
            self.terminals.update(set(s.remove_structure()))
            
        # Construct rules and non-terminals
        sentence_list = list(self.sentences)
        number = 1
        for sent in sentence_list:
            number = self.expand('S',sent.structure,number)
            
        # Compute weights
        for key,value in self.rules.items():
            self.weights[key] = [1/len(self.rules[key]) for i in range(1,len(self.rules[key])+1)]
            
        self.grammar = ProbabilisticGrammar(self.terminals, self.non_terminals, self.rules, self.weights)
        
    
    def expand(self,symb,structure,number):
        e1 = structure[0]
        e2 = structure[1]
        self.non_terminals.add(symb)
        if symb not in self.rules:
            self.rules[symb] = []
        if (type(e1) != list) and (type(e2) != list):
            self.rules[symb].append([e1,e2])
        elif (type(e1) == list) and (type(e2) != list):
            symbol = 'S'+str(number)
            number += 1
            self.rules[symb].append([symbol,e2])
            number = self.expand(symbol,e1,number)
            # expand symbol with e1
        elif (type(e1) != list) and (type(e2) == list):
            symbol = 'S'+str(number)
            number += 1

            self.rules[symb].append([e1,symbol])
            number = self.expand(symbol,e2,number)

            # expand symbol with e2
        elif (type(e1) == list) and (type(e2) == list):
            symbol1 = 'S'+str(number)

            number += 1
            symbol2 = 'S'+str(number)

            number += 1
            self.rules[symb].append([symbol1,symbol2])
            number = self.expand(symbol1,e1,number)

            number = self.expand(symbol2,e2,number)

            # expand symbol1 with e1 and expand symbol2 with e2
        return number
    
    def generate_sentence(self):
        if self.grammar != None:
            return self.grammar.generate_sentence('S')
    
