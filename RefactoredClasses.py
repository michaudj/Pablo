# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:11:07 2025

@author: jemi6917
"""



import copy
import re
import numpy as np

import random


from itertools import accumulate

from dataclasses import dataclass

def change_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = new_value

class Type:
    # create a cache to store instances of the Type class
    cache = {}
    prim_ID = 1
    
    # override the __new__ method to check the cache for an existing instance
    def __new__(cls, formula):
        
        # check if the key is in the cache
        if formula in cls.cache:
            # if the key is in the cache, return the corresponding instance
            return cls.cache[formula]
        else:
            # if the key is not in the cache, create a new instance
            instance = super().__new__(cls)
            # store the new instance in the cache
            cls.cache[formula] = instance
            # return the new instance
            return instance
    
    # initialize the instance with its formula
    def __init__(self, formula):
        self.formula = formula

    @staticmethod
    def create_primitive_type():
        tt = Type( str(Type.prim_ID))
        Type.prim_ID += 1
        return tt
        
    def __hash__(self):
        return hash(frozenset(self.formula))
        
    # define the string representation of the instance
    def __repr__(self):
        string = self.formula.replace('o','/')
        string = string.replace('u','\\')
        return string    
    
    def __eq__(self, other):
        return self is other
    
    def split(self,pu=0.5,prim='New',bad_s1=None,bad_s2=None): # should return two types that combine into the initial type
        print(bad_s1)
        if prim == None:
            prim_type = Type('0')
        elif prim == 'New':
            pass
        else:
            if not prim.is_primitive():
                prim = 'New'
            else:
                prim_type = prim
                print(prim_type)
        print(prim)
            
        if random.random() < pu:
            if prim == prim:
                if bad_s1 != None:
                    if prim in bad_s1:
                        prim = 'New'
            if prim == 'New':
                if bad_s1 != None:
                    index = 0
                    while Type(str(index)) in bad_s1:
                        index += 1
                    prim_type = Type(str(index))
                elif bad_s2 != None:
                    prim_type = Type("0")
                else:
                    prim_type = Type('0')#self.create_primitive_type()
            return [prim_type, Type(prim_type.formula+"u"+self.formula )]
        else:
            if prim == prim:
                if bad_s2 != None:
                    if prim in bad_s2:
                        prim = 'New'
                    
            if prim == 'New':
                if bad_s2 != None:
                    index = 0
                    while Type(str(index)) in bad_s2:
                        index += 1
                    prim_type = Type(str(index))
                elif bad_s1 != None:
                    prim_type = Type('0')
                else:
                    prim_type = Type('0')
            return [Type(self.formula+"o"+prim_type.formula ), prim_type]
        pass
    
    def is_start(self): # Checks whether the type is expecting something on the left. 
        return len(self.left_compatible_chunks()) == 0
    
    def get_primitives(self):
        return re.split(r"u|o",self.formula)
    
    
    def left_compatible_chunks(self):
        #substrings = re.findall(r".*?u", self.formula)
        substrings = re.findall(r"^"+self.left_type()+"u",self.formula)
        substrings = list(accumulate(substrings))
        #print(substrings)
        substrings = [re.sub(r"u$", "$", x) for x in substrings]
        #print(substrings)
        substrings1 = [re.sub(r"^", r"^", x) for x in substrings]
        #print(substrings1)
        substrings2 = [re.sub(r"^", r"u", x) for x in substrings]
        #print(substrings2)
        return substrings1 + substrings2
    
    def right_compatible_chunks(self):
        #substrings = re.findall(r".*?o", self.formula[::-1])
        substrings = re.findall(r"o"+self.right_type()+"$",self.formula)
        substrings = list(accumulate(substrings))
        substrings = [re.sub(r"^o", "", x) for x in substrings]
        substrings1 = [re.sub(r"^", r"^", x) for x in substrings]
        substrings2 = [re.sub(r"$", r"o", x) for x in substrings1]
        substrings1 = [re.sub(r"$", r"$", x) for x in substrings1]
        return substrings1 + substrings2
    
    def is_empty(self):
        return len(self.get_primitives()) == 1 and len(self.get_primitives()[0])==0
    
    
    def is_primitive(self):
        if len(self.get_primitives()) == 1 and len(self.get_primitives()[0])!=0:
            return True
        else:
            return False
        #return (len(self.right_compatible_chunks()) + len(self.left_compatible_chunks())) == 0
    
    def left_type(self):
        primitives = self.get_primitives()
        return primitives[0]
    
    def right_type(self):
        primitives = self.get_primitives()
        return primitives[-1]
    
    def is_right_compatible(self, other):
        for i, pattern in enumerate(self.right_compatible_chunks()):
            #print(pattern)
            match = re.search(pattern, other.formula)
            if match:
                return True, pattern
        return False, None
    
    def is_left_compatible(self, other):
        for i, pattern in enumerate(other.left_compatible_chunks()):
            #print(pattern)
            match = re.search(pattern, self.formula)
            if match:
                return True, pattern
        return False, None
    
    def is_compatible(self, other):
        return self.is_left_compatible(other)[0] or self.is_right_compatible(other)[0]

    
    def __add__(self, other):
        if self.is_left_compatible(other)[0]:
            #print('left_compatible')
            pattern = self.is_left_compatible(other)[1][:-1]
            if pattern.startswith("^"):
                l = len(pattern)
                return Type(other.formula[l:])
            elif pattern.startswith("u"):
                l = len(pattern)
                return Type(self.formula[:-l+1]+other.formula[l:])
            else:
                print('Problem here.')
        elif self.is_right_compatible(other)[0]:
            #print('right_compatible')
            pattern = self.is_right_compatible(other)[1][1:]
            if pattern.endswith("$"):
                l = len(pattern)
                return Type(self.formula[:-l])
            elif pattern.endswith("o"):
                l = len(pattern)
                return Type(self.formula[:-l+1]+other.formula[l:])
            else:
                print('Problem here.')
        else:
            raise TypeError("Incompatible types")
            
    @staticmethod
    def reduce(types):
        remaining_types = types[:]
        
        # keep trying to reduce the list of types until it contains only one type
        while len(remaining_types) > 1:
            # set the reduced flag to False
            reduced = False
            
            # iterate over the remaining types
            for i, type1 in enumerate(remaining_types[:-1]):
                type2 = remaining_types[i+1]
                # if the two types are compatible, reduce them and update the reduced flag
                if type1.is_compatible(type2):
                    remaining_types[i] = type1 + type2
                    del remaining_types[i+1]
                    reduced = True
                    break
        
            # return the remaining type
            if not reduced:
                return remaining_types
        
        return remaining_types
    
    @staticmethod
    def is_sentence(types):
        remaining_types = Type.reduce(types)
        if len(remaining_types) == 1 and remaining_types[0] == Type('0'):
            return True
        else:
            return False

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
        def modify_element_at_depth(nested_list, depth, new_value):
            for i in range(depth-1):
                nested_list = nested_list[-1]
            nested_list[-1] = [nested_list[-1],new_value]
            
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
        def flatten(lst):
            flat_list = []
            for item in lst:
                if isinstance(item, list):
                    flat_list.extend(flatten(item))
                else:
                    flat_list.append(item)
            return flat_list
        
        if type(self.structure) is not list:
            return self.structure
        else:
            return flatten(self.structure)
    

class TChunk(Chunk):
    
    def __init__(self, structure):
        super().__init__(structure)
        
    def chunk_at_depth(self, other, depth=0):
        def modify_element_at_depth(nested_list, depth, new_value):
            for i in range(depth-1):
                nested_list = nested_list[-1]
            nested_list[-1] = [nested_list[-1],new_value]
            
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
    def __init__(self, learner,initial_value_border = 1.0, initial_value_chunking=-1):
        self.behaviour_repertoire = {}
        
    def update_repertoire(self,couple): # couple must be a couple of SChunks
        #print('call of update repertoire')
        substantial_couple = (str(couple[0]),str(couple[1]))
        if substantial_couple not in self.behaviour_repertoire:
            values = [self.initial_value_border]
            values += [self.initial_value_chunking for i in range(couple[0].get_depth()+1)]
            self.behaviour_repertoire[substantial_couple] =np.array(values)# np.array([Learner.initial_value_border] + [Learner.initial_value_chunking for i in range(couple[0].depth+1)])
        if substantial_couple not in self.couple_str_to_couple:
            self.couple_str_to_couple[substantial_couple] = couple    
    
    def add(self,chunk):
        pass
    pass

class WorkingMemory:
    """Class that specifies how the input is processed. Include functionality for
    loading stimulus in working memory, choosing behaviour based on the behavioural repertoire and
    calling the reinforcement engine when needed."""
    
    def __init__(self,learner):
        self.learner = learner
        self.events = []
        self.typing_events = []
        self.border_before = False
        self.border_within = False
        
    
    def process(self,stimuli_stream,s1,s2_index):
        s2 = Chunk(stimuli_stream.read_stimulus(s2_index))
        response = self.choose_behaviour((s1,s2)) #inside this function should the wm events be updated
        self.events.append(((s1,s2),response))
        
        if response == 0: #border
            # Test success condition
            # If success update learning history and reinforce decision
            # If failure update learning history and reinforce negatively
            pass
        else: # Chunking
            # do the chunking at the correct level
            pass
        pass
    
    def choose_behaviour(self, pair):
        pass
    
    def clear(self):
        self.events = []
        self.typing_events = []
        self.border_before = False
        self.border_within = False
    

class LearningHistory:
    """Class to store data on successes and failure for plotting purpose..."""
    def __init__(self,learner):
        self.n_reinf = 0
        self.learner = learner
        self.success = []
        self.sent_len = []
        
    def update_history(self,success,length):
        self.n_reinf += 1
        self.success.append(success)
        self.sent_len.append(length)
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
    
    def __init__(self, learner,alpha, beta,positive_reinforcement = 1.0, negative_reinforcement = -1):
        self.learner = learner
        self.alpha = alpha
        self.beta = beta
        self.positive_reinforcement = positive_reinforcement
        self.negative_reinforcement = negative_reinforcement

    
    def reinforce(self, reinforcement = 'positive'):
        def get_sub_couples( couple):
            sub_pairs = []
            for s in couple[0].get_right_subchunks(couple[0].get_depth()):
                sub_pairs.append((s,couple[1]))
                #self.update_repertoire((s,couple[1]))
            return sub_pairs
        
        #print('call of reinforce')
        # for each events reinforce behaviour associated to chunk
        if reinforcement == 'positive':
            u = Learner.positive_reinforcement
        elif reinforcement == 'negative':
            u = Learner.negative_reinforcement
        #print(self.events)
        

        
        for couple,r in self.learner.wm.events:
            #substantial_couple= (str(couple[0]),str(couple[1]))
            #print('reinforcement')
            #Q = self.behaviour_repertoire[substantial_couple][r]
            subevents=[(couple,r)]
            subpairs = get_sub_couples(couple)
            for pair in subpairs:
                substantial_pair = (str(pair[0]),str(pair[1]))
                self.learner.ltm.update_repertoire(pair)
                if r < len(self.ltm.behaviour_repertoire[substantial_pair]):
                    subevents.append((pair,r))
                    # Q += self.behaviour_repertoire[substantial_pair][r]
                    #print(subevents)
            for p,rr in subevents:
                substantial_p = (str(p[0]),str(p[1]))
                self.ltm.behaviour_repertoire[substantial_p][rr] += self.alpha * (u - self.learner.ltm.behaviour_repertoire[substantial_p][rr])


class Learner():
    
    ID = 0
    
    def __init__(self, learner_type='flexible', border_type='next'):
        self.ID = Learner.ID + 1
        Learner.ID +=1
        self.type = learner_type
        self.border_type = border_type
        self.ltm = LongTermMemory(self)
        self.wm = WorkingMemory(self)
        self.reinforcer = ReinforcementEngine(self)
        self.history = LearningHistory(self)
        self.grammar = None
        
    def __repr__(self):
        return f"Learner {self.ID}"
        
    def learn(self, stimuli_stream, num_trials):
        s1 = Chunk(stimuli_stream.read_stimulus(0))
        # Try to type here?
        # add s1 to working memory and long term memory
        self.ltm.add(s1)
        
        
        s2_index = 1
        
        while self.history.n_reinf <= num_trials:
            s1, s2_index = self.wm.process(stimuli_stream,s1,s2_index)
        
        