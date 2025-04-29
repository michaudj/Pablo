# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:11:07 2025

@author: jemi6917
"""


import copy
import re

from dataclasses import dataclass

class Chunk():
    
    cache = {}
    
        # override the __new__ method to check the cache for an existing instance
    def __new__(cls, structure):
        key = hash(frozenset(str(structure)))

        # check if the key is in the cache
        if key in cls.cache:
            # if the key is in the cache, return the corresponding instance
            return cls.cache[key]
        else:
            # if the key is not in the cache, create a new instance
            instance = super().__new__(cls)
            # store the new instance in the cache
            cls.cache[key] = instance
            # return the new instance
            return instance
    
    def __init__(self, structure):
        self.structure = structure
        self.depth = self.get_depth()
        
    def __repr__(self):
        return str(self.structure)
    
    def __hash__(self):
        return hash(frozenset(str(self.structure)))
    
    def get_s1(self):
        return Chunk(self.structure[0])
    
    def get_s2(self):
        return Chunk(self.structure[1])
    
    def get_right_subchunks(self, depth):
        right_subchunks = []
        nested_list = self.structure[:]
        for d in range(depth):
            nested_list = nested_list[-1]
            right_subchunks.append(Chunk(nested_list))
        return right_subchunks
    
    def chunk_at_depth(self, other, depth=0):
        if type(self.structure)!= list:
            nested_list = copy.deepcopy(self.structure)
        else:
            nested_list = self.structure[:]
        
        if depth == 0:
            struct = [nested_list,other.structure]
            return Chunk(struct)
        else:
            modify_element_at_depth(nested_list, depth, other.structure)
            return Chunk(nested_list)
    
    def get_depth(self):
        st = str(self.structure)
        match = re.search("]*$",st)
        return len(match.group(0))
    
    def remove_structure(self):
        if type(self.structure) is not list:
            return self.structure
        else:
            return flatten(self.structure)
    

class TChunk(Chunk):
    
    def __init__(self, structure):
        super().__init__(structure)
        
    def chunk_at_depth(self, other, depth=0):
        if type(self.structure)!= list:
            nested_list = self.structure
        else:
            nested_list = list(self.structure)
        
        if depth == 0:
            return TChunk([nested_list,other.structure])
        else:
            modify_element_at_depth(nested_list, depth, other.structure)
            return TChunk(nested_list)
        
    def change_element_at_depth(self, element, depth=0):
        if type(self.structure)!= list:
            nested_list = self.structure
        else:
            nested_list = list(self.structure)
        
        if depth == 0:
            return TChunk(element)
        else:
            change_element_at_depth(nested_list, depth, element)
            return TChunk(nested_list)
        
    def is_sentence(self):
        if type(self.structure)!= list:
            if self.structure == Type('0'):
                return True
            else:
                return False
        elif type(self.structure) == list:
            remaining_type = reduce_types(self.remove_structure()[:])
            if len(remaining_type) == 1 and remaining_type[0] == Type('0'):
                return True
            else:
                return False
        else:
            return False
        
    def reduce(self):
        if type(self.structure) != list:
            return self.structure
        else:
            [s1,s2] = self.structure[:]
            s1 = TChunk(s1)
            s2 = TChunk(s2)
            if type(s1.structure) == Type and type(s2.structure)== Type:
                result = s1.structure + s2.structure
                return result
            elif type(s1.structure) != Type and type(s2.structure)== Type:
                result = s1.reduce() + s2.structure
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = TChunk([s1.structure,s2.structure])
                return result
            elif type(s1.structure) == Type and type(s2.structure)!= Type:
                result = s1.structure + s2.reduce()
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = TChunk([s1.structure,s2.structure])
                return result
            else:
                t1 = s1.reduce()
                t2 = s2.reduce()
                result = t1 + t2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = TChunk([s1.structure,s2.structure])

                return result
                    
        
    def is_consistent(self):
        if type(self.structure) != list:
            return True
        else:
            [s1,s2] = self.structure[:]
            s1 = TChunk(s1)
            s2 = TChunk(s2)

            if type(s1.structure) == Type and type(s2.structure)== Type:
                return s1.structure.is_compatible(s2.structure)
            elif type(s1.structure) != Type and type(s2.structure)== Type:
                if s1.is_consistent():
                    self = TChunk([s1.structure,s2.structure])
                    return s1.reduce().is_compatible(s2.structure)
                else:
                    return False
            elif type(s1.structure) == Type and type(s2.structure)!= Type:
                if s2.is_consistent():
                    self = TChunk([s1.structure,s2.structure])
                    return s1.structure.is_compatible(s2.reduce())
                else:
                    return False
            else:
                if s1.is_consistent() and s2.is_consistent():
                    self = TChunk([s1.structure,s2.structure])
                    return s1.reduce().is_compatible(s2.reduce())
                else:
                    return False
       
    def right_types(self):
        # Only works if TChunk is consistent!!!
        list_of_reduced_types = [self.reduce()]
        
        for chunk in self.get_right_subchunks(self.depth):
            list_of_reduced_types.append(chunk.reduce())
        return list_of_reduced_types
            
class VChunk(Chunk):
    def __init__(self, structure):
        super().__init__(structure)
        
    def chunk_at_depth(self, other, depth=0):
        if type(self.structure)!= list:
            nested_list = self.structure
        else:
            nested_list = self.structure[:]
        
        if depth == 0:
            return VChunk([nested_list,other.structure])
        else:
            modify_element_at_depth(nested_list, depth, other.structure)
            return VChunk(nested_list)
        
    def average(self):
        if type(self.structure) != list:
            return self.structure
        else:
            [s1,s2] = self.structure[:]
            s1 = VChunk(s1)
            s2 = VChunk(s2)
            if type(s1.structure) != list and type(s2.structure)!= list:
                result = (s1.structure + s2.structure)/2
                return result
            elif type(s1.structure) == list and type(s2.structure)!= list:
                result = (s1.average() + s2.structure)/2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = VChunk([s1.structure,s2.structure])
                return result
            elif type(s1.structure) != list and type(s2.structure) == list:
                result = (s1.structure + s2.average())/2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = VChunk([s1.structure,s2.structure])
                return result
            else:
                t1 = s1.average()
                t2 = s2.average()
                result = (t1 + t2)/2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = VChunk([s1.structure,s2.structure])

                return result
            
    def right_values(self):
        # Only works if TChunk is consistent!!!
        list_of_reduced_types = [self.average()]
        
        for chunk in self.get_right_subchunks(self.depth):
            list_of_reduced_types.append(chunk.average())
        return list_of_reduced_types

@dataclass 
class LearnerConfig:
    alpha: float = 0.2
    beta: float = 1.0
    positive_reinforcement: float = 5.0
    negative_reinforcement: float = -1.0
    initial_value_chunking: float = -1.0
    initial_value_border: float = 1.0
    initial_value_type: float = 1.0
    good_type_value: float = 1.0
    bad_type_value: float = 0.0
    
class TypeAssigner:
    """Class to handle type assignement"""
    pass 

class LongTermMemory:
    """Class that handles everything that is stored in long term memory.
    In particular, it contains the behavioralRepertoire of the learner,
    i.e. pairs of chunks associated with the values of their behavior."""
    pass

class WorkingMemory:
    """Class that specifies how the input is processed. Include functionality for
    loading stimulus in working memory, choosing behaviour based on the behavioural repertoire and
    calling the reinforcement engine when needed."""
    
    def __init__(self):
        self.events = []
        self.typing_events = []
        pass
    pass

class LearningHistory:
    """Class to store data on successes and failure for plotting purpose..."""
    def __init__(self):
        self.n_reinf = 0
        self.success = []
        self.sent_len = []
    pass

class Grammar: #### Consider using the PCFG class for this...
    """Class to extract grammatical information from the long term memory"""
    def __init__(self):
        self.sentences = set()
        self.terminals = set()
        self.non_terminals = set()
        self.rules = dict()
        self.weights = dict()
    pass

class ReinforcementEngine:
    """Class to implement various types of reinforcement learning algorithms"""
    pass


class Learner():
    
    ID = 0
    
    def __init__(self, learner_type='flexible', border_type='next'):
        self.ID = Learner.ID + 1
        Learner.ID +=1
        self.ltm = LongTermMemory()
        self.wm = WorkingMemory()
        self.reinforcer = ReinforcementEngine()
        self.history = LearningHistory()
        
    def __repr__(self):
        return f"Learner {self.ID}"
        
    def learn(self, stimuli_stream, num_trials):
        s1 = Chunk(stimuli_stream.read_stimulus(0))
        
        pass
        