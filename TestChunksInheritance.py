# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:26:00 2023

@author: jmd01
"""

#import json
from copy import deepcopy
import re
import numpy as np
import random
from new_raw_input import ProbabilisticGrammar

from itertools import accumulate
import types
#import sys
#sys.setrecursionlimit(1500)



def modify_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = [nested_list[-1],new_value]
    
def change_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = new_value
    
def change_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = new_value
    
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def add_weights(b_values1,b_values2):
    a = b_values1[:]
    b = b_values2[:]
    l = sorted((a, b), key=len)
    c = l[1].copy()
    c[:len(l[0])] += l[0]
    return c

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
        if prim == None:
            prim_type = Type('0')
        elif prim == 'New':
            pass
        else:
            if not prim.is_primitive():
                prim = 'New'
            else:
                prim_type = prim

            
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

class Stimuli():
    
    def __init__(self,substantial,t='',v=0):
        self.content = substantial
        self.type = t
        self.value = v
        
    def __repr__(self):
        return str(self.content)# + ', t:'+str(self.type)+', v:'+str(self.value)+')'
    
    def longstr(self):
        return '('+str(self.content) + ', t:'+str(self.type)+', v:'+str(self.value)+')'

    
    def retype(self,new_type,new_value):
        self.type = new_type
        self.value = new_value
        
    def copy(self):
        return Stimuli(self.content,t=self.type,v=self.value)
        
class SChunk():

    def __init__(self, structure):
        self.structure = structure
        self.depth = self.get_depth()
        
    def __repr__(self):
        return str(self.structure)
    
    def __hash__(self):
        return hash(frozenset((self.structure)))
    
    def get_s1(self):
        return SChunk(self.structure[0])
    
    def get_s2(self):
        return SChunk(self.structure[1])
    
    
    def get_right_subchunks(self, depth):
        right_subchunks = []
        nested_list = deepcopy(self.structure)
        for d in range(depth):
            nested_list = nested_list[-1]
            right_subchunks.append(SChunk(nested_list))
        return right_subchunks
    
    def chunk_at_depth(self, other, depth=0):
        nested_list = deepcopy(self.structure)
        # if type(self.structure)!= list:
        #     nested_list = self.structure
        # else:
        #     #nested_list = copy.deepcopy(self.structure)
        #     nested_list = list(self.structure)
        
        if depth == 0:
            struct = [nested_list,other.structure]
            return SChunk(struct)
        else:
            modify_element_at_depth(nested_list, depth, other.structure)
            return SChunk(nested_list)
    
    def get_depth(self):
        st = str(self.structure)
        match = re.search("]*$",st)
        return len(match.group(0))
    
    def remove_structure(self):
        if type(self.structure) is not list:
            return [self.structure]
        else:
            return flatten(self.structure)
        
    def get_list_types(self):
        return [Type(self.remove_structure()[i].type) for i in range(len(self.remove_structure()))]
    
    def get_list_values(self):
        return [self.remove_structure()[i].value for i in range(len(self.remove_structure()))]
        
    def reduce(self):
        if type(self.structure) != list:
            return Type(self.structure.type)
        else:
            [s1,s2] = self.structure[:]
            s1 = SChunk(s1)
            s2 = SChunk(s2)
            if type(s1.structure) != list and type(s2.structure) != list:
                result = Type(s1.structure.type) + Type(s2.structure.type)
                return result
            elif type(s1.structure) == list and type(s2.structure)!= list:
                result = s1.reduce() + Type(s2.structure.type)
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = SChunk([s1.structure,s2.structure])
                return result
            elif type(s1.structure) != list and type(s2.structure) == list:
                result = Type(s1.structure.type) + s2.reduce()
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = SChunk([s1.structure,s2.structure])
                return result
            else:
                t1 = s1.reduce()
                t2 = s2.reduce()
                result = t1 + t2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = SChunk([s1.structure,s2.structure])

                return result  
            
    def is_consistent(self):
        if type(self.structure) != list:
            return True
        else:
            [s1,s2] = self.structure[:]
            s1 = SChunk(s1)
            s2 = SChunk(s2)

            if type(s1.structure) != list and type(s2.structure) != list:
                return Type(s1.structure.type).is_compatible(Type(s2.structure.type))
            elif type(s1.structure) == list and type(s2.structure)!= list:
                if s1.is_consistent():
                    self = SChunk([s1.structure,s2.structure])
                    return s1.reduce().is_compatible(Type(s2.structure.type))
                else:
                    return False
            elif type(s1.structure) != list and type(s2.structure)== list:
                if s2.is_consistent():
                    self = SChunk([s1.structure,s2.structure])
                    return Type(s1.structure.type).is_compatible(s2.reduce())
                else:
                    return False
            else:
                if s1.is_consistent() and s2.is_consistent():
                    self = SChunk([s1.structure,s2.structure])
                    return s1.reduce().is_compatible(s2.reduce())
                else:
                    return False
                
    def right_types(self):
        # Only works if TChunk is consistent!!!
        list_of_reduced_types = [self.reduce()]
        
        for chunk in self.get_right_subchunks(self.get_depth()):
            list_of_reduced_types.append(chunk.reduce())
        return list_of_reduced_types
    
    def average(self):
        if type(self.structure) != list:
            return self.structure.value
        else:
            [s1,s2] = self.structure[:]
            s1 = SChunk(s1)
            s2 = SChunk(s2)
            if type(s1.structure) != list and type(s2.structure)!= list:
                result = (s1.structure.value + s2.structure.value)/2
                return result
            elif type(s1.structure) == list and type(s2.structure)!= list:
                result = (s1.average() + s2.structure.value)/2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = SChunk([s1.structure,s2.structure])
                return result
            elif type(s1.structure) != list and type(s2.structure) == list:
                result = (s1.structure.value + s2.average())/2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = SChunk([s1.structure,s2.structure])
                return result
            else:
                t1 = s1.average()
                t2 = s2.average()
                result = (t1 + t2)/2
                # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
                self = SChunk([s1.structure,s2.structure])

                return result
            
    def right_values(self):
        # Only works if TChunk is consistent!!!
        list_of_reduced_types = [self.average()]
        
        for chunk in self.get_right_subchunks(self.get_depth()):
            list_of_reduced_types.append(chunk.average())
        return list_of_reduced_types
    
    def right_types_dict(self):
        if self.is_consistent():
            return dict(zip(self.right_types(),self.right_values()))
        else:
            return None


####################################################
####################################################
####################################################
####################################################
####################################################



class Learner():
    
    alpha = 0.2
    beta = 1.
    positive_reinforcement = 5.
    negative_reinforcement = -1.
    initial_value_chunking = -1.
    initial_value_border = 1.
    
    initial_value_type = 1.
    good_type_value = 1.
    bad_type_value = 0.
    
    ID = 0
    
    def __init__(self, n_trials = 14, border = 'next'):
        self.ID = Learner.ID + 1
        Learner.ID +=1
        self.border_type = border # or 'default'
        self.n_trials = n_trials
        self.n_reinf = 0
        self.success = []
        self.sent_len = []
        self.sentences = set()
        # Change this set into a dictionary where the key is a dictionary of types
        # self.chunks = set()
        
        self.behaviour_repertoire = {} # dictionary of where the keys are couples of chunks and the value a list of behavioural values
        self.events = [] # encodes the current list of couples ((chunk,chunk), behaviour) to reinforce
        self.stimuli = []
        self.decisions = []
        
        self.border_before = True
        self.border_within = False
        
        # for grammar extraction
        self.terminals = set()
        self.non_terminals = set()
        self.rules = dict()
        self.weights = dict()
        self.grammar = None
        
        self.s1_typed = False
        self.stimulus_types_dict = dict() #keys are strings represented a single stimulus and values are dictionary with types and values associated to them.
        self.typatory = dict() # keys are types and values are related to how good the type is
        self.typing_events = [] # encodes the current list of typings (format to be defined)
        self.typed_structure = None # This will hold the current typing of the sentence. From it, types to be reinforced will be recovert.
        
        
    def __repr__(self):
        string = 'Learner ' + str(self.ID)
        
        return string

     
    #Not needed for flexible learner
    def add_chunk(self, chunk): # chunk must be a string
        if chunk not in self.stimulus_types_dict:
            self.stimulus_types_dict[chunk] = {}
            
    def update_stimulus_type_association(self,s):
        if type(s.structure) != list:
            self.add_chunk(str(s))
            
    def update_repertoire(self,couple): # couple must be a couple of SChunks
        #print('call of update repertoire')
        substantial_couple = (str(couple[0]),str(couple[1]))
        if substantial_couple not in self.behaviour_repertoire:
            #print('repertoire initialization')
            values = [Learner.initial_value_border]
            values += [Learner.initial_value_chunking for i in range(couple[0].get_depth()+1)]
            self.behaviour_repertoire[substantial_couple] =np.array(values)# np.array([Learner.initial_value_border] + [Learner.initial_value_chunking for i in range(couple[0].depth+1)])

                
    def get_sub_couples(self, couple):
        sub_pairs = []
        for s in couple[0].get_right_subchunks(couple[0].get_depth()):
            self.update_stimulus_type_association(s)

            sub_pairs.append((s,couple[1]))
            self.update_repertoire((s,couple[1]))
        return sub_pairs
    

    def learn(self,stimuli_stream):
        # initialize stimuli
        s1 = SChunk(stimuli_stream.stimuli[0])
        self.stimuli.append(s1)
        #self.typed_structure = TChunk('')
        #self.s1_typed = False
        
        #self.chunks.add(s1)
        self.update_stimulus_type_association(s1)
        s2_index = 1
        #for t in range(self.n_trials):
        while self.n_reinf <= self.n_trials:
            s1, s2_index = self.respond(stimuli_stream, s1, s2_index)

            
    
    def respond(self,stimuli_stream,s1,s2_index):
        # get the s2 stimuli and make it a chunk
        s2 = SChunk(stimuli_stream.stimuli[s2_index])
        self.stimuli.append(s2)
        # update chunkatory
        self.update_stimulus_type_association(s2)
        

        response = self.choose_behaviour((s1,s2))
        #print(self.behaviour_repertoire[(str(s1),str(s2))])
        self.decisions.append(response)

        self.events.append(((s1,s2),response))
        
                
        if response == 0: # boundary placement
            #print('Border')
            # increment the number of reinforcement events by 1
            self.n_reinf += 1 
            # check if border is correctly placed
            is_border = stimuli_stream.border_before[s2_index]

            if is_border and not self.border_within and self.border_before:
                #print('Good Unit')
                # perform positive reinforcement
                # Store sentence (not cognitively plausible but used for grammar extraction)
                self.sentences.add(str(s1))

                #print(self.typing_events)
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
                new_s1 = SChunk(new_s1)
                # self.typed_structure = TChunk('')
                # self.s1_typed = False
            else:
                self.border_before = stimuli_stream.border_before[s2_index]
                new_s1,s2_index = s2, s2_index + 1
                # self.typed_structure = TChunk('')
                # self.s1_typed = False
                
            #self.chunks.add(new_s1)
            self.update_stimulus_type_association(new_s1)

            self.border_within = False
            
            self.stimuli.append(new_s1)
            

            
        else: # some type of chunking occurs
            # Check if there was a border
            if not self.border_within:
                self.border_within = stimuli_stream.border_before[s2_index]
            
            # Perform chunking at correct level
            new_s1 = s1.chunk_at_depth(s2,depth=s1.get_depth()+1-response) 
            s2_index+=1
            self.update_stimulus_type_association(new_s1)

            
        return new_s1, s2_index
    
    def choose_behaviour(self,couple):
        substantial_couple = (str(couple[0]),str(couple[1]))
        self.update_repertoire(couple)
        b_range = len(self.behaviour_repertoire[substantial_couple])
        options = [i for i in range(b_range)]
        z = deepcopy(self.behaviour_repertoire[substantial_couple])


        subpairs = self.get_sub_couples(couple)
        
        norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
        # Accumulate support from subchunks
        for pair in subpairs:
            substantial_pair = (str(pair[0]),str(pair[1]))
            lenp = len(self.behaviour_repertoire[substantial_pair])
            z[:lenp] += self.behaviour_repertoire[substantial_pair]
            # Take the average
        z /= norm_vec
        weights = np.exp(Learner.beta * z)

            
        response = random.choices(options,weights/np.sum(weights))
        return response[0]  
    

    def reinforce(self, reinforcement = 'positive'):
        #print('call of reinforce')
        # for each events reinforce behaviour associated to chunk
        if reinforcement == 'positive':
            u = Learner.positive_reinforcement
        elif reinforcement == 'negative':
            u = Learner.negative_reinforcement
        
        #for e in self.events:
        #    print(e)
        for couple,r in self.events:
            substantial_couple= (str(couple[0]),str(couple[1]))
            #print('reinforcement')
            Q = self.behaviour_repertoire[substantial_couple][r]
            subevents = [(couple,r)]
            subpairs = self.get_sub_couples(couple)
            for pair in subpairs:
                substantial_pair = (str(pair[0]),str(pair[1]))
                self.update_repertoire(pair)
                if r < len(self.behaviour_repertoire[substantial_pair]):
                    subevents.append((pair,r))
                    Q += self.behaviour_repertoire[substantial_pair][r]
            #Q /= len(subevents)

        for p,rr in subevents:
            substantial_p = (str(p[0]),str(p[1]))
            self.behaviour_repertoire[substantial_p][rr] += Learner.alpha * (u - Q)

                
        
        # Clear working memory
        self.events = []
        self.stimuli = []
        self.decisions = []

        
    
    # def get_types_s1(self,s1):
    #     return self.typed_structure.right_types()
    
    # def get_values_s1(self,s1):
    #     pass
       
    
    # def assign_type_new2(self,s1,s2):
    #     if not self.typed_structure.is_consistent():
    #         self.s1_typed = False
    #     if self.s1_typed:
    #         #print('s1 typed')
    #         # That function should return a dictionary of reduced types from self.typed_structure, with their value
    #         types = self.get_types_s1()
    #         values = self.get_values_s1()
    #         types_s1 = dict()
    #         success = False
    #         t1 = self.choose_type_from_dict(types_s1)
    #         default = t1
    #         while not success and len(types_s1)!= 0:
    #             success, t2 = self.compatible_t2_new2(s2)
    #             if not success:
    #                 del types_s1[t1]
    #                 if len(types_s1)!=0:
    #                     t1 = self.choose_type_from_dict(types_s1)
    #         if not success:
    #             t1 = default
    #             t2 = self.assign_t2(t1,s2)
    #         return t2
            
    #     else:
    #         #print('s1 not typed')
    #         #print(self.typed_structure)
    #         #print(s1)
    #         # NEEDS TO BE CHANGED: Jointly look through both typings using merge_dict...
    #         good_starting_types = self.good_starting_types(s1)
    #         good_types_s2 = self.good_types(s2)
            
            
            
    #         #print(good_starting_types)
    #         if good_starting_types != None and good_types_s2 == None:
    #             #print('s1 get typed')
    #             success = False
    #             t1 = self.choose_type_from_dict(good_starting_types)
    #             default = t1
    #             # Look for compatible t2
    #             while not success and len(good_starting_types) !=0:
    #                 success, t2 = self.compatible_t2_new2(s2)
    #                 if not success:
    #                     del good_starting_types[t1]
    #                     if len(good_starting_types) != 0:
    #                         t1 = self.choose_type_from_dict(good_starting_types)
    #             if not success:
    #                 t1 = default
    #                 t2 = self.assign_t2(t1,s2)
                    
    #             self.typed_structure = TChunk(t1)
    #             #print(t1)
    #             self.s1_typed = True
    #             return t2
    #         elif good_starting_types == None and good_types_s2 != None:
    #             success = False
    #             t2 = self.choose_type_from_dict(good_types_s2)
    #             default = t2
    #             while not success and len(good_types_s2) !=0:
    #                 success, t1 = self.compatible_t1_new2(s1)
    #                 if not success:
    #                     del good_types_s2[t2]
    #                     if len(good_types_s2)!=0:
    #                         t2 = self.choose_type_from_dict(good_types_s2)
    #             if not success:
    #                 t2 = default
    #                 t1 = self.assign_t1(t2,s1)
                
    #             # Need to check if s1 is complex use split otherwise update self.typed_structure
    #             self.s1_typed = True
    #             return t2
            
    #         elif good_starting_types != None and good_types_s2 != None:
    #             merge = self.merge_dicts(good_starting_types, good_types_s2)
    #             success = False
    #             (index,t) = self.choose_type_from_dict(merge)
    #             default = (t,index)
    #             while not success and len(merge)!=0:
    #                 if index ==1:
    #                     pass
    #                 elif index == 2:
    #                     pass
    #                 else:
    #                     print('Problem')
    #                 if not success:
    #                     del merge[(index,t)]
    #                     if len(merge)!= 0:
    #                         (index,t) = self.choose_type_from_dict(merge)
    #             if not success:
    #                 if default[0]==1:
    #                     pass
    #                 elif default[0]==2:
    #                     pass
    #                 else:
    #                     print('Problem')
                        
    #             self.s1_typed = True
    #             return t2
                
    #         else:   
    #         #print(self.typed_structure)
    #             #print('No typing')
    #             #print(self.typed_structure)
    #             self.s1_typed = False
    #             return Type('')
            
    # def compatible_t2_new2(self,s2):
    #     # Need to return success and t2
    #     structure = self.typed_structure.remove_structure()
    #     #print(type(self.typed_structure.remove_structure()))
    #     if type(structure) == Type:
    #         t1 = structure
    #     else:
    #         # This needs to be updated
    #         # Not working well for longer structures
    #         # Should take into account the full structure of the TChunk
    #         t1 = self.typed_structure.remove_structure()[-1]
    #     #print(type(t1))
    #     if s2 in self.chunk_dict:
    #         possible_t2 = dict()
    #         for t,v in self.chunk_dict[s2].items():
    #             if t1.is_compatible(t) and v >= Learner.bad_type_value:
    #                 tt = t1 + t
    #                 if tt.is_start():
    #                     possible_t2[t] = v
    #         if len(possible_t2) == 0:
    #             return None
    #         else:
    #             return possible_t2
    #     else:
    #         return None
        
    # def compatible_t1_new2(self,s2):
    #     structure = self.typed_structure.remove_structure()
    #     #print(type(self.typed_structure.remove_structure()))
    #     if type(structure) == Type:
    #         t1 = structure
    #     else:
    #         # This needs to be updated
    #         # Not working well for longer structures
    #         # Should take into account the full structure of the TChunk
    #         t1 = self.typed_structure.remove_structure()[-1]
    #     #print(type(t1))
    #     if s2 in self.chunk_dict:
    #         possible_t2 = dict()
    #         for t,v in self.chunk_dict[s2].items():
    #             if t1.is_compatible(t) and v >= Learner.bad_type_value:
    #                 tt = t1 + t
    #                 if tt.is_start():
    #                     possible_t2[t] = v
    #         if len(possible_t2) == 0:
    #             return None
    #         else:
    #             return possible_t2
    #     else:
    #         return None
        
#########################################################
#
#       Will become obsolete
#
#########################################################
    
    # def assign_type_new(self,s1,s2):
    #     if self.s1_typed:
    #         #print('s1 typed')
    #         #print(self.typed_structure)
    #         #print(s1)
    #         #print(self.typed_structure)
    #         compatible_t2 = self.compatible_t2_new(s2)
    #         #print(compatible_t2)
    #         if compatible_t2 != None:
    #             #print('found compatible')
    #             t2 = self.choose_type_from_dict(compatible_t2)
    #             #self.typing_events.append((s2,t2))
    #             #print('t2: ' +str(t2))
    #         else:
    #             # Create new type for t2
    #             #print('no compatible')
    #             t2 = self.assign_new_type_to_s2_new(s2)
    #         #print('t2 is chosen')
    #         #print(t2)
    #         return t2
    #         # Look for compatible types for s2
    #     else:
    #         #print('s1 not typed')
    #         #print(self.typed_structure)
    #         #print(s1)
    #         # NEEDS TO BE CHANGED: Jointly look through both typings using merge_dict...
    #         good_starting_types = self.good_starting_types(s1)
    #         #good_types_s2 = self.good_types(s2)
            
            
            
    #         #print(good_starting_types)
    #         if good_starting_types != None:# and good_types_s2 == None:
    #             #print('s1 get typed')
    #             # Rest needs to be rewritten
    #             t1 = self.choose_type_from_dict(good_starting_types)
    #             self.typed_structure = TChunk(t1)
    #             #print(t1)
    #             self.s1_typed = True
    #             compatible_t2 = self.compatible_t2_new(s2)
    #             #print('Any compatible t2')
    #             #print(compatible_t2)
    #             if compatible_t2 != None:
    #                 #print('found compatible')
    #                 t2 = self.choose_type_from_dict(compatible_t2)
    #                 #self.typing_events.append((s2,t2))
    #                 #print('t2: ' +str(t2))
    #             else:
    #                 # Create new type for t2
    #                 #print('no compatible')
    #                 t2 = self.assign_new_type_to_s2_new(s2)
    #             #print('s2 has been typed too')
    #             #print(t2)
    #             return t2

    #         else:   
    #         #print(self.typed_structure)
    #             #print('No typing')
    #             #print(self.typed_structure)
    #             self.s1_typed = False
    #             return Type('')
            
            
    # def compatible_t2_new(self,s2):
    #     structure = self.typed_structure.remove_structure()
    #     #print(type(self.typed_structure.remove_structure()))
    #     if type(structure) == Type:
    #         t1 = structure
    #     else:
    #         # This needs to be updated
    #         # Not working well for longer structures
    #         # Should take into account the full structure of the TChunk
    #         t1 = self.typed_structure.remove_structure()[-1]
    #     #print(type(t1))
    #     if s2 in self.chunk_dict:
    #         possible_t2 = dict()
    #         for t,v in self.chunk_dict[s2].items():
    #             if t1.is_compatible(t) and v >= Learner.bad_type_value:
    #                 tt = t1 + t
    #                 if tt.is_start():
    #                     possible_t2[t] = v
    #         if len(possible_t2) == 0:
    #             return None
    #         else:
    #             return possible_t2
    #     else:
    #         return None
        
    # def assign_new_type_to_s2_new(self,s2):
    #     structure = self.typed_structure.remove_structure()
    #     if type(structure) == Type:
    #         t1 = structure
    #     else:
    #         t1 = self.typed_structure.remove_structure()[-1]
    #     #print('t1 is ')
    #     #print(t1)
    #     #print(t1.is_primitive())
    #     if self.bad_types(s2) == None:
    #         #print('no bad types')
    #         if t1.is_primitive():
    #             if t1 == Type('0'):
    #                 t2 = Type('0')
    #             else:
    #                 t2 = Type(t1.formula + 'u0')
                
    #             #print('t2: ' +str(t2))
    #         else: # complex type
    #             #print('t1 complex, fullfil expectation')
    #             #print(t1)
    #             if len(t1.right_compatible_chunks())==0:
    #                 #print('not expecting after')
    #                 if Type(t1.formula[-1]) == Type('0'):
    #                     t2 = Type('0')
    #                 else:
    #                     t2 = Type(t1.formula[-1]+'u0')
    #             else:
    #                 #print('expecting after')
    #                 t2 = Type(t1.formula[-1])
    #         #self.associate_type_to_chunk(s2, t2)    
    #     else:
    #         #print('there are bad types')
    #         #print(self.bad_types(s2))
    #         if t1.is_primitive():
    #             if t1 == Type('0'):
    #                 index = 0
    #                 while Type(str(index)) in self.bad_types(s2):
    #                     index +=1
    #                 t2 = Type(str(index))
    #             else:
    #                 #print('primitive, but not sentence')
    #                 index = 0
    #                 while Type(t1.formula + 'u'+str(index)) in self.bad_types(s2):
    #                     index += 1
    #                 t2 = Type(t1.formula  + 'u' + str(index))
                
    #             #print('t2: ' +str(t2))
    #         else:
    #             #print('t1 complex, fullfill expectation if possible')
    #             #print(t1)
    #             if len(t1.right_compatible_chunks())==0:
    #                 #print('not expecting after')
    #                 if Type(t1.formula[-1]) == Type('0'):
    #                     index = 0
    #                     while Type(str(index)) in self.bad_types(s2):
    #                         index +=1
    #                     t2 = Type(str(index))
    #                 else:
    #                     t2 = Type(t1.formula[-1]+'u0')
                    
    #             else:
    #                 #print('expecting after')
    #                 #t2 = Type(t1.formula[-1])
    #                 index = 0
                    
    #                 while Type(str(index)) in self.bad_types(s2):
    #                     index +=1
    #                 change_type = Type(t1.formula[-1]+'o'+str(index))
    #                 t2 = Type(str(index))
                    
    #                 # Remove the old typing
    #                 #self.typing_events.remove((s1,t1))
    #                 #print('Retyping, not implemented')
    #                 #print('old t1')
    #                 #print(t1)
    #                 #print(self.typed_structure)
    #                 #print(t2)
    #                 #print('t1 should be')
    #                 #print('change_type')
    #                 #print(t1+change_type)
    #                 #print(t1+change_type)
    #                 # IF MORE THAN 10 PRIMITIVES: ERROR
    #                 self.typed_structure = self.typed_structure.change_element_at_depth( t1+change_type,depth=self.typed_structure.depth)
    #                 #print(self.typed_structure)
                    
    #                 # Change self.typed_structure to propagate retyping event
    #                 #t1 =  t1 + change_type
    #                 #self.associate_type_to_chunk(s1, t1)
    #     #self.associate_type_to_chunk(s2, t2)
    #     return t2
            


        
    # def good_types(self,chunk):
    #     if chunk in self.chunk_dict:
    #         good_types = dict()
    #         for t,v in self.chunk_dict[chunk].items():
    #             if v >= Learner.good_type_value:
    #                 good_types[t] = v
    #         if len(good_types) == 0:
    #             return None
    #         else:
    #             return good_types
    #     else:
    #         return None
    
        
    # def bad_types(self,chunk):
    #     if chunk in self.chunk_dict:
    #         good_types = dict()
    #         for t,v in self.chunk_dict[chunk].items():
    #             if v <= Learner.bad_type_value:
    #                 good_types[t] = v
    #         if len(good_types) == 0:
    #             return None
    #         else:
    #             return good_types
    #     else:
    #         return None
        
    # def good_starting_types(self,chunk):
    #     good_types = self.good_types(chunk)
    #     if good_types != None:
    #         good_start_types = dict()
    #         for t,v in good_types.items():
    #             if t.is_start():
    #                 good_start_types[t]=v
    #         if len(good_start_types) == 0:
    #             return None
    #         else:
    #             return good_start_types
    #     else:
    #         return None
            
        
    # def show_good_types(self):
    #     for key, dic in self.chunk_dict.items():
    #         gt = self.good_types(key)
    #         if gt != None:
    #             print(key)
    #             print(gt)
                
    # def show_bad_types(self):
    #     for key, dic in self.chunk_dict.items():
    #         gt = self.bad_types(key)
    #         if gt != None:
    #             print(key)
    #             print(gt)
                
    # def show_good_starting_types(self):
    #     for key, dic in self.chunk_dict.items():
    #         gst = self.good_starting_types(key)
    #         if gst != None:
    #             print(key)
    #             print(gst)



            
    # def associate_type_to_chunk(self,chunk,t):
    #     self.typing_events.append((chunk,t))
    #     if t not in self.typatory:
    #         self.typatory[t] = {chunk}
    #     else:
    #         self.typatory[t].add(chunk)
    #     if t not in self.chunk_dict[chunk]:
    #         self.chunk_dict[chunk][t] = Learner.initial_value_type
            

    
    # def merge_dicts(self,dic1,dic2):
    #     merge_dict = dict()
    #     for t,v in dic1.items():
    #         merge_dict[(1,t)] = v
    #     for t,v in dic2.items():
    #         merge_dict[(2,t)] = v
    #     return merge_dict
    
    # def choose_type_from_dict(self,dico):
    #     keys = list(dico.keys())
    #     values = np.array(list(dico.values()))
    #     weights = np.exp(Learner.beta * values)
    #     response = random.choices(keys,weights/np.sum(weights))
    #     #print(response)
    #     return response[0]
    
    
    
    # def type_chunk(self,s1,type_to_split):
    #     # Update typing event here
    #     #self.typing_events.append((s1,type_to_split))
    #     structure = s1.remove_structure()
    #     types = dict()
    #     if type(structure)==str:
    #         # extract possible types
    #         types[structure] = self.good_types(s1)
    #         pass
    #     else:
    #         # extract possible types by looping over the element of structure
    #         for s in structure:
    #             types[s] = self.good_types(Chunk(s))
    #         pass
    #     # Check if any of the elements are typed
    #     anytypes = False
    #     for key,value in types.items():
    #         if value != None:
    #             anytypes = True
    #             break
        
    #     if not anytypes:
    #         if type(s1.structure) is not str:
    #             new_s1 = s1.get_s1()
    #             new_s2 = s1.get_s2()
    #             [t1,t2] = type_to_split.split()
    #             self.type_chunk(new_s1,t1)
    #             self.type_chunk(new_s2,t2)
    #         else:
    #             self.associate_type_to_chunk(s1, type_to_split)
                
    #     else:
    #         # Check from bottom up if there is a possible typing
    #         if type(s1.structure) is not str:
    #             new_s1 = s1.get_s1()
    #             new_s2 = s1.get_s2()
    #             # Check if s1 and/or s2 are primitive
    #             if type(new_s1.structure) == str and type(new_s2.structure) == str:
    #                 #print('both are primitive')
    #                 # 1 typed, 2 typed, both typed?
    #                 if types[new_s1.structure]==None and types[new_s2.structure]!=None:
    #                     #print('Case 1')
    #                     success = False
    #                     t2 = self.choose_type_from_dict(types[new_s2.structure])
    #                     default = t2
    #                     while not success and len(types[new_s2.structure])!=0:
    #                         success, t1 = self.choose_compatible_t1(t2,new_s1,type_to_split)
    #                         if not success:
    #                             del types[new_s2.structure][t2]
    #                             if len(types[new_s2.structure])!=0:
    #                                 t2 = self.choose_type_from_dict(types[new_s2.structure])
    #                     if not success:
    #                         # POSSIBLY NEW TYPES
    #                         [t1,t2] = type_to_split.split(pu=0,prim='New')
    #                     self.associate_type_to_chunk(new_s1, t1)
    #                     self.associate_type_to_chunk(new_s2, t2)
        
    #                 elif types[new_s1.structure]!=None and types[new_s2.structure]==None:
    #                     #print('Case 2')
    #                     success = False
    #                     t1 = self.choose_type_from_dict(types[new_s1.structure])
    #                     default = t1
    #                     while not success and len(types[new_s1.structure])!=0:
    #                         success, t2 = self.choose_compatible_t2(t1,new_s2,type_to_split)
    #                         if not success:
    #                             del types[new_s1.structure][t1]
    #                             if len(types[new_s1.structure])!=0:
    #                                 t1 = self.choose_type_from_dict(types[new_s1.structure])
    #                     if not success:
    #                         # POSSIBLY NEW TYPES
    #                         [t1,t2] = type_to_split.split(pu=1,prim='New')
    #                     self.associate_type_to_chunk(new_s1, t1)
    #                     self.associate_type_to_chunk(new_s2, t2)
    #                 else:
    #                     #print('Case 3')
    #                     success = False
    #                     merge = self.merge_dicts(types[new_s1.structure], types[new_s2.structure])
    #                     (index,t) = self.choose_type_from_dict(merge)
    #                     default = (t,index)
    #                     while not success and len(merge) != 0:
    #                         if index == 1:
    #                             t1 = t
    #                             success, t2 = self.choose_compatible_t2(t1,new_s2,type_to_split)
    #                             if not success:
    #                                 del merge[(index,t)]
    #                                 if len(merge) != 0:
    #                                     (index,t) = self.choose_type_from_dict(merge)
    #                         elif index ==2:
    #                             t2 = t
    #                             success, t1 = self.choose_compatible_t1(t2,new_s1,type_to_split)
    #                             if not success:
    #                                 del merge[(index,t)]
    #                                 if len(merge) != 0:
    #                                     (index,t) = self.choose_type_from_dict(merge)
    #                         else:  
    #                             print('Problem here')
    #                     if not success:
    #                         # Possibly new types
    #                         if default[1]==1:
    #                             t1 = default[0]
    #                             [t1,t2] = type_to_split.split(pu=1,prim='New',bad_s1=self.bad_types(new_s1))
                                
    #                         elif default[1]==2:
    #                             t2 = default[0]
    #                             [t1,t2] = type_to_split.split(pu=0,prim='New',bad_s2 = self.bad_types(new_s2))
    #                         else:
    #                             print('Problem here')
    #                     self.associate_type_to_chunk(new_s1, t1)
    #                     self.associate_type_to_chunk(new_s2, t2)       
    #                     # Merge dict, choose dominant, loop through
    #                     pass
    #             elif type(new_s1.structure) == str and type(new_s2.structure) != str:
    #                 #print('s1 primitive, s2 complex')
    #                 # where are the typings
    #                 #print(types)
    #                 #print(new_s1)
    #                 if types[new_s1.structure] != None:
    #                     # success = False
    #                     t1 = self.choose_type_from_dict(types[new_s1.structure])
    #                     [t1,t2] = type_to_split.split(pu=1,prim=t1)
    #                     self.associate_type_to_chunk(new_s1, t1)
    #                     self.type_chunk(new_s2,t2)
    #                 else:
    #                     [t1,t2] = type_to_split.split()
    #                     self.associate_type_to_chunk(new_s1,t1)
    #                     self.type_chunk(new_s2,t2)
    #             elif type(new_s2.structure) == str and type(new_s1.structure) != str:
    #                 #print('s2 primitive, s1 complex')
    #                 if types[new_s2.structure] != None:
    #                     # success = False
    #                     #print('Case 1')
    #                     t2 = self.choose_type_from_dict(types[new_s2.structure])
    #                     [t1,t2] = type_to_split.split(pu=0,prim=t2)
    #                     #print(t1)
    #                     #print(t2)
    #                     self.associate_type_to_chunk(new_s2, t2)
    #                     #print(self.typing_events)
    #                     self.type_chunk(new_s1,t1)
    #                 else:
    #                     #print('Case 2')
    #                     [t1,t2] = type_to_split.split()
    #                     self.associate_type_to_chunk(new_s2,t2)
    #                     self.type_chunk(new_s1,t1)
    #             else:
    #                 #print('both are complex')
    #                 [t1,t2] = type_to_split.split()
    #                 self.type_chunk(new_s2,t2)
    #                 self.type_chunk(new_s1,t1)
            

        
    # def choose_compatible_t1(self,t2,s1,type_to_split):
    #     success = True
    #     bad_types= self.bad_types(s1)
    #     if bad_types == None:
    #         if t2.is_primitive():
    #             [t1,t2]=type_to_split.split(pu=0,prim=t2)
    #             success = True
    #         else:
    #             if len(t2.left_compatible_chunks())==0:
    #                 #print('not expecting before')
    #                 success = False
    #                 t1 = None
    #             else:
    #                 #print('expecting before')
    #                 t1 = Type(t2.formula[0])
    #                 if t1.is_compatible(t2):
    #                     if t1+t2 == type_to_split:
    #                         success = True
    #                     else:
    #                         success = False
    #                         t1 = None
    #                 else:
    #                     success = False
    #                     t1 = None
    #     else:
    #         if t2.is_primitive():
    #             [t1,t2]=type_to_split.split(pu=0,prim=t2)
    #             if t1 in self.bad_types(s1):
    #                 success = False
    #                 t1 = None
    #             else:
    #                 success = True

    #         else:
    #             if len(t2.left_compatible_chunks())==0:
    #                 success = False
    #                 t1 = None
    #             else:
    #                 t1 = Type(t2.formula[0])
    #                 if t1.is_compatible(t2) and t1 not in self.bad_types(s1):
    #                     if t1+t2 == type_to_split:
    #                         success = True
    #                     else:
    #                         success = False
    #                         t1 = None
    #                 else:
    #                     success = False
    #                     t1 = None
    #     return success, t1


    # def choose_compatible_t2(self,t1,s2,type_to_split):
    #     success = True
    #     bad_types= self.bad_types(s2)
    #     if bad_types == None:
    #         if t1.is_primitive():
    #             [t1,t2]=type_to_split.split(pu=1,prim=t1)
    #             success = True
    #         else:
    #             if len(t1.right_compatible_chunks())==0:
    #                 #print('not expecting before')
    #                 success = False
    #                 t2 = None
    #             else:
    #                 #print('expecting before')
    #                 t2 = Type(t1.formula[-1])
    #                 if t1.is_compatible(t2):
    #                     if t1+t2 == type_to_split:
    #                         success = True
    #                     else:
    #                         success = False
    #                         t1 = None
    #                 else:
    #                     success = False
    #                     t1 = None
    #     else:
    #         if t1.is_primitive():
    #             [t1,t2]=type_to_split.split(pu=1,prim=t1)
    #             if t1 in self.bad_types(s2):
    #                 success = False
    #                 t1 = None
    #             else:
    #                 success = True

    #         else:
    #             if len(t1.right_compatible_chunks())==0:
    #                 success = False
    #                 t2 = None
    #             else:
    #                 t2 = Type(t1.formula[-1])
    #                 if t1.is_compatible(t2) and t2 not in self.bad_types(s2):
    #                     if t1+t2 == type_to_split:
    #                         success = True
    #                     else:
    #                         success = False
    #                         t2 = None
    #                 else:
    #                     success = False
    #                     t2 = None
    #     return success, t2   


        
print('-------------------------------------')
print('-------     Test      ---------------')
print('-------------------------------------')

s1 = Stimuli('n1','',0)
s2 = Stimuli('v1','',0)
s3 = Stimuli('n2','',0)
s4 = Stimuli('n1','',0)
s5 = Stimuli('n1','',0)
s6 = Stimuli('v1','',0)
s7 = Stimuli('n2','',0)
s8 = Stimuli('n1','',0)
stimulis = [s1,s2,s3,s4]
print([stimulis[i].content for i in range(len(stimulis))])
type_association = dict(zip([stimulis[i].content for i in range(len(stimulis))],[stimulis[i].type for i in range(len(stimulis))]))

c1 = SChunk(s1)
c2 = SChunk(s2)
c3 = c1.chunk_at_depth(c2)
c4 = SChunk(s3)
c5 = c3.chunk_at_depth(c4,depth=1)

c6 = SChunk(s5)
c7 = SChunk(s6)
c8 = c6.chunk_at_depth(c7)
c9 = SChunk(s7)
c10 = c8.chunk_at_depth(c9,depth=1)
print(c1)
print(c2)
print(c3)
print(c5)

print('test equality')
print(c1.structure.content==c6.structure.content)
c1.structure.content = 'n10'
print(c1.structure.content==c6.structure.content)

print(c5.get_list_types())
print(c5.get_list_values())

ttypes = ['1','1u0o1','1']
values = [1,2,3]


print("retyping")
structure = c5.remove_structure()
for i in range(len(structure)):
    structure[i].retype(ttypes[i],values[i])
    
print(c5)

print(c5.get_list_types())
print(c5.get_list_values())
print(c5.reduce())

print('test right subchunks')
rc = c5.get_right_subchunks(c5.depth)
print(rc)
rc[-1].structure.content = 'n12'
print(rc)

print(c5.structure)




