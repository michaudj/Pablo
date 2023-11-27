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
#from Raw_input import ProbabilisticGrammar

from itertools import accumulate
import pandas as pd

import sys
sys.setrecursionlimit(1500)



def modify_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = [nested_list[-1],new_value]
    
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

# class Type:
#     # create a cache to store instances of the Type class
#     cache = {}
#     prim_ID = 1
    
#     # override the __new__ method to check the cache for an existing instance
#     def __new__(cls, formula):
        
#         # check if the key is in the cache
#         if formula in cls.cache:
#             # if the key is in the cache, return the corresponding instance
#             return cls.cache[formula]
#         else:
#             # if the key is not in the cache, create a new instance
#             instance = super().__new__(cls)
#             # store the new instance in the cache
#             cls.cache[formula] = instance
#             # return the new instance
#             return instance
    
#     # initialize the instance with its formula
#     def __init__(self, formula):
#         self.formula = formula
        

#     @staticmethod
#     def create_primitive_type():
#         tt = Type( str(Type.prim_ID))
#         Type.prim_ID += 1
#         return tt
        
#     def __hash__(self):
#         return hash(frozenset(self.formula))
        
#     # define the string representation of the instance
#     def __repr__(self):
#         string = self.formula.replace('o','/')
#         string = string.replace('u','\\')
#         return string    
    
#     def __eq__(self, other):
#         return self is other
    
#     def __len__(self):
#         return len(self.get_primitives())
    
#     def split(self,pu=0.5,prim='New',bad_s1=None,bad_s2=None): # should return two types that combine into the initial type
#         if prim == None:
#             prim_type = Type('0')
#         elif prim == 'New':
#             pass
#         else:
#             if not prim.is_primitive():
#                 prim = 'New'
#             else:
#                 prim_type = prim

            
#         if random.random() < pu:
#             if prim == prim:
#                 if bad_s1 != None:
#                     if prim in bad_s1:
#                         prim = 'New'
#             if prim == 'New':
#                 if bad_s1 != None:
#                     index = 0
#                     while Type(str(index)) in bad_s1:
#                         index += 1
#                     prim_type = Type(str(index))
#                 elif bad_s2 != None:
#                     prim_type = Type("0")
#                 else:
#                     prim_type = Type('0')#self.create_primitive_type()
#             return [prim_type, Type(prim_type.formula+"u"+self.formula )]
#         else:
#             if prim == prim:
#                 if bad_s2 != None:
#                     if prim in bad_s2:
#                         prim = 'New'
                    
#             if prim == 'New':
#                 if bad_s2 != None:
#                     index = 0
#                     while Type(str(index)) in bad_s2:
#                         index += 1
#                     prim_type = Type(str(index))
#                 elif bad_s1 != None:
#                     prim_type = Type('0')
#                 else:
#                     prim_type = Type('0')
#             return [Type(self.formula+"o"+prim_type.formula ), prim_type]
#         pass
    
#     def is_start(self): # Checks whether the type is expecting something on the left. 
#         return len(self.left_compatible_chunks()) == 0
    
#     def get_primitives(self):
#         return re.split(r"u|o",self.formula)
    
#     def is_expecting_before(self):
#         if len(self.left_compatible_chunks()) !=  0:
#             return True
#         else:
#             return False
        
#     def is_expecting_after(self):
#         if len(self.right_compatible_chunks()) !=  0:
#             return True
#         else:
#             return False
    
    
#     def left_compatible_chunks(self):
#         #substrings = re.findall(r".*?u", self.formula)
#         substrings = re.findall(r"^"+self.left_type()+"u",self.formula)
#         substrings = list(accumulate(substrings))
#         #print(substrings)
#         substrings = [re.sub(r"u$", "$", x) for x in substrings]
#         #print(substrings)
#         substrings1 = [re.sub(r"^", r"^", x) for x in substrings]
#         #print(substrings1)
#         substrings2 = [re.sub(r"^", r"u", x) for x in substrings]
#         #print(substrings2)
#         return substrings1 + substrings2
    
#     def right_compatible_chunks(self):
#         #substrings = re.findall(r".*?o", self.formula[::-1])
#         substrings = re.findall(r"o"+self.right_type()+"$",self.formula)
#         substrings = list(accumulate(substrings))
#         substrings = [re.sub(r"^o", "", x) for x in substrings]
#         substrings1 = [re.sub(r"^", r"^", x) for x in substrings]
#         substrings2 = [re.sub(r"$", r"o", x) for x in substrings1]
#         substrings1 = [re.sub(r"$", r"$", x) for x in substrings1]
#         return substrings1 + substrings2
    
#     def is_empty(self):
#         return len(self.get_primitives()) == 1 and len(self.get_primitives()[0])==0
    
    
#     def is_primitive(self):
#         if len(self.get_primitives()) == 1 and len(self.get_primitives()[0])!=0:
#             return True
#         else:
#             return False
#         #return (len(self.right_compatible_chunks()) + len(self.left_compatible_chunks())) == 0
    
#     def left_type(self):
#         primitives = self.get_primitives()
#         return primitives[0]
    
#     def right_type(self):
#         primitives = self.get_primitives()
#         return primitives[-1]
    
#     def is_right_compatible(self, other):
#         for i, pattern in enumerate(self.right_compatible_chunks()):
#             #print(pattern)
#             match = re.search(pattern, other.formula)
#             if match:
#                 return True, pattern
#         return False, None
    
#     def is_left_compatible(self, other):
#         for i, pattern in enumerate(other.left_compatible_chunks()):
#             #print(pattern)
#             match = re.search(pattern, self.formula)
#             if match:
#                 return True, pattern
#         return False, None
    
#     def is_compatible(self, other):
#         return self.is_left_compatible(other)[0] or self.is_right_compatible(other)[0]

    
#     def __add__(self, other):
#         if self.is_left_compatible(other)[0]:
#             #print('left_compatible')
#             pattern = self.is_left_compatible(other)[1][:-1]
#             if pattern.startswith("^"):
#                 l = len(pattern)
#                 return Type(other.formula[l:])
#             elif pattern.startswith("u"):
#                 l = len(pattern)
#                 return Type(self.formula[:-l+1]+other.formula[l:])
#             else:
#                 print('Problem here.')
#         elif self.is_right_compatible(other)[0]:
#             #print('right_compatible')
#             pattern = self.is_right_compatible(other)[1][1:]
#             if pattern.endswith("$"):
#                 l = len(pattern)
#                 return Type(self.formula[:-l])
#             elif pattern.endswith("o"):
#                 l = len(pattern)
#                 return Type(self.formula[:-l+1]+other.formula[l:])
#             else:
#                 print('Problem here.')
#         else:
#             raise TypeError("Incompatible types")
            
#     # @staticmethod
#     # def reduce(types):
#     #     remaining_types = types[:]
        
#     #     # keep trying to reduce the list of types until it contains only one type
#     #     while len(remaining_types) > 1:
#     #         # set the reduced flag to False
#     #         reduced = False
            
#     #         # iterate over the remaining types
#     #         for i, type1 in enumerate(remaining_types[:-1]):
#     #             type2 = remaining_types[i+1]
#     #             # if the two types are compatible, reduce them and update the reduced flag
#     #             if type1.is_compatible(type2):
#     #                 remaining_types[i] = type1 + type2
#     #                 del remaining_types[i+1]
#     #                 reduced = True
#     #                 break
        
#     #         # return the remaining type
#     #         if not reduced:
#     #             return remaining_types
        
#     #     return remaining_types
    
#     # @staticmethod
#     # def is_sentence(types):
#     #     remaining_types = Type.reduce(types)
#     #     if len(remaining_types) == 1 and remaining_types[0] == Type('0'):
#     #         return True
#     #     else:
#     #         return False

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
        
    def get_type(self):
        return Type(self.type)
        
    # def copy(self):
    #     return Stimuli(self.content,t=self.type,v=self.value)
        
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
    
    def __len__(self):
        return len(self.remove_structure())
    
    
    def get_right_subchunks(self, depth):
        right_subchunks = []
        nested_list = deepcopy(self.structure)
        for d in range(depth):
            nested_list = nested_list[-1]
            right_subchunks.append(SChunk(nested_list))
        return right_subchunks
    
    def chunk_at_depth(self, other, depth=0):
        nested_list = deepcopy(self.structure)
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
        
    # def get_list_types(self):
    #     return [self.remove_structure()[i].get_type() for i in range(len(self.remove_structure()))]
    
    # def get_list_values(self):
    #     return [self.remove_structure()[i].value for i in range(len(self.remove_structure()))]
    
    # def get_list_content(self):
    #     return [self.remove_structure()[i].content for i in range(len(self.remove_structure()))]
    
    # def get_typing_events(self):
    #     return list(zip(self.get_list_content(),self.get_list_types()))
        
    # def reduce(self):
    #     if type(self.structure) != list:
    #         return self.structure.get_type()
    #     else:
    #         [s1,s2] = self.structure[:]
    #         s1 = SChunk(s1)
    #         s2 = SChunk(s2)
    #         if type(s1.structure) != list and type(s2.structure) != list:
    #             result = s1.structure.get_type() + s2.structure.get_type()
    #             return result
    #         elif type(s1.structure) == list and type(s2.structure)!= list:
    #             result = s1.reduce() + s2.structure.get_type()
    #             # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
    #             self = SChunk([s1.structure,s2.structure])
    #             return result
    #         elif type(s1.structure) != list and type(s2.structure) == list:
    #             result = s1.structure.get_type() + s2.reduce()
    #             # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
    #             self = SChunk([s1.structure,s2.structure])
    #             return result
    #         else:
    #             t1 = s1.reduce()
    #             t2 = s2.reduce()
    #             result = t1 + t2
    #             # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
    #             self = SChunk([s1.structure,s2.structure])

    #             return result  
            
    # def get_sub_types(self):
    #     return self.get_s1().reduce(), self.get_s2().reduce()
    
            
    # def is_consistent(self):
    #     if type(self.structure) != list:
    #         if self.structure.get_type().is_empty():
    #             return False
    #         else:
    #             return True
    #     else:
    #         [s1,s2] = self.structure[:]
    #         s1 = SChunk(s1)
    #         s2 = SChunk(s2)

    #         if type(s1.structure) != list and type(s2.structure) != list:
    #             return s1.structure.get_type().is_compatible(s2.structure.get_type())
    #         elif type(s1.structure) == list and type(s2.structure)!= list:
    #             if s1.is_consistent():
    #                 self = SChunk([s1.structure,s2.structure])
    #                 return s1.reduce().is_compatible(s2.structure.get_type())
    #             else:
    #                 return False
    #         elif type(s1.structure) != list and type(s2.structure)== list:
    #             if s2.is_consistent():
    #                 self = SChunk([s1.structure,s2.structure])
    #                 return s1.structure.get_type().is_compatible(s2.reduce())
    #             else:
    #                 return False
    #         else:
    #             if s1.is_consistent() and s2.is_consistent():
    #                 self = SChunk([s1.structure,s2.structure])
    #                 return s1.reduce().is_compatible(s2.reduce())
    #             else:
    #                 return False
                
    # # def retype_at_position(self,index,t_formula,t_value):
    # #     self.remove_structure()[index].retype(t_formula,t_value)
        
    # def get_type_list_from_new_expectation(self,t_old,t_new):
    #     index = self.index_right_type_expect_after(t_old)
    #     mod_type = Type(self.right_types()[index].right_type()+'o'+t_new.formula)
    #     type_to_split = self.right_types()[index] + mod_type
    #     list_types = SChunk.create_list_of_types(self.right_chunks()[index], type_to_split)
    #     new_types = [t for t in self.get_list_types()]
    #     len_chunk = len(self.right_chunks()[index])
    #     new_types[-len_chunk:] = list_types[:]
    #     return new_types
    
    # def retype_from_expectation(self,t_old,t_new,dico):
    #     list_of_types = self.get_type_list_from_new_expectation(t_old, t_new)
    #     list_of_values = []
    #     for i in range(len(list_of_types)):
    #         if list_of_types[i].formula in dico:
    #             list_of_values.append(dico[list_of_types[i].formula])
    #             new_index = None
    #         else:
    #             new_index = i
    #             list_of_values.append(TypeLearner.initial_value_type)
    #     #list_of_values = [dico[list_of_types[i].formula] for i in range(len(list_of_types))]
        
    #     for i in range(len(self.remove_structure())):
    #         if self.remove_structure()[i].type != list_of_types[i]:
    #             self.remove_structure()[i].retype(list_of_types[i].formula,list_of_values[i])
    #     if new_index:
    #         return SChunk(self.remove_structure()[new_index]), list_of_types[new_index]
    #     else:
    #         return None
    
    # @staticmethod
    # def create_list_of_types(chunk,t):
    #     if type(chunk.structure) != list:
    #         return [t]
    #     else:
    #         s1 = chunk.get_s1()
    #         s2 = chunk.get_s2()
    #         sub_types = chunk.get_sub_types()
    #         if sub_types[0].is_expecting_after():
    #             type_list = t.split(pu=0,prim=sub_types[1])
    #         elif sub_types[1].is_expecting_before():
    #             type_list = t.split(pu=1,prim=sub_types[0])
    #         if type(s1.structure) != list and type(s2.structure) != list:
    #             return type_list
    #         elif type(s1.structure) == list and type(s2.structure) != list:
    #             return SChunk.create_list_of_types(s1, type_list[0]) + [type_list[1]]
    #         elif type(s1.structure) != list and type(s2.structure) == list:
    #             return type_list[0] + SChunk.create_list_of_types(s2, type_list[1])
    #         elif type(s1.structure) == list and type(s2.structure) == list:
    #             return SChunk.create_list_of_types(s1, type_list[0]) + SChunk.create_list_of_types(s2, type_list[1])
                
    # def right_types(self):
    #     # Only works if TChunk is consistent!!!
    #     list_of_reduced_types = [self.reduce()]
        
    #     for chunk in self.get_right_subchunks(self.get_depth()):
    #         list_of_reduced_types.append(chunk.reduce())
    #     return list_of_reduced_types
    
    # def index_right_type_expect_after(self,tt):
    #     indices = []
    #     i=0
    #     for t in self.right_types():
    #         if t.is_expecting_after():
    #             if t.right_type() == tt.formula:
    #                 indices.append(i)
    #         i += 1
    #     if len(indices) == 0:
    #         return None
    #     else:
    #         return max(indices)
    
    def right_chunks(self):
        return [self]+self.get_right_subchunks(self.depth)
    
    def right_len(self):
        return [len(c.remove_structure()) for c in self.right_chunks()]
    
    # def average(self):
    #     if type(self.structure) != list:
    #         return self.structure.value
    #     else:
    #         [s1,s2] = self.structure[:]
    #         s1 = SChunk(s1)
    #         s2 = SChunk(s2)
    #         if type(s1.structure) != list and type(s2.structure)!= list:
    #             result = (s1.structure.value + s2.structure.value)/2
    #             return result
    #         elif type(s1.structure) == list and type(s2.structure)!= list:
    #             result = (s1.average() + s2.structure.value)/2
    #             # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
    #             self = SChunk([s1.structure,s2.structure])
    #             return result
    #         elif type(s1.structure) != list and type(s2.structure) == list:
    #             result = (s1.structure.value + s2.average())/2
    #             # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
    #             self = SChunk([s1.structure,s2.structure])
    #             return result
    #         else:
    #             t1 = s1.average()
    #             t2 = s2.average()
    #             result = (t1 + t2)/2
    #             # Weird bug fixed by the following line: if more than one element reduce to 0, creates bug...
    #             self = SChunk([s1.structure,s2.structure])

    #             return result
            
    # def right_values(self):
    #     # Only works if TChunk is consistent!!!
    #     list_of_reduced_types = [self.average()]
        
    #     for chunk in self.get_right_subchunks(self.get_depth()):
    #         list_of_reduced_types.append(chunk.average())
    #     return list_of_reduced_types
    
    # def right_types_dict(self):
    #     if self.is_consistent():
    #         return dict(zip(self.right_types(),self.right_values()))
    #     else:
    #         return None


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
        
        
    def __repr__(self):
        return 'Learner ' + str(self.ID)

            
    def update_repertoire(self,couple): # couple must be a couple of SChunks
        #print('call of update repertoire')
        substantial_couple = (str(couple[0]),str(couple[1]))
        if substantial_couple not in self.behaviour_repertoire:
            values = [Learner.initial_value_border]
            values += [Learner.initial_value_chunking for i in range(couple[0].get_depth()+1)]
            self.behaviour_repertoire[substantial_couple] =np.array(values)# np.array([Learner.initial_value_border] + [Learner.initial_value_chunking for i in range(couple[0].depth+1)])

                
    def get_sub_couples(self, couple):
        sub_pairs = []
        for s in couple[0].get_right_subchunks(couple[0].get_depth()):

            sub_pairs.append((s,couple[1]))
            self.update_repertoire((s,couple[1]))
        return sub_pairs
    

    def learn(self,stimuli_stream):
        # initialize stimuli
        s1 = SChunk(stimuli_stream.stimuli[0])
        s2_index = 1
        #for t in range(self.n_trials):
        while self.n_reinf <= self.n_trials:
            s1, s2_index = self.respond(stimuli_stream, s1, s2_index)
            # under condition save snapshot to file

    def learn_with_snapshot(self,stimuli_stream,filename,snap_times,threshold):
        # initialize stimuli
        s1 = SChunk(stimuli_stream.stimuli[0])
        s2_index = 1
        #for t in range(self.n_trials):
        with pd.ExcelWriter(filename) as f:
            dff = pd.DataFrame([(self.alpha, self.beta,self.positive_reinforcement,self.negative_reinforcement,self.initial_value_border,self.initial_value_chunking)],columns=['alpha','beta','positive reinforcement','negative reinforcement','initial value border','initial value chunking'])
            dff.to_excel(f,sheet_name='parameters')
            while self.n_reinf <= self.n_trials:
                s1, s2_index = self.respond(stimuli_stream, s1, s2_index)
                if self.n_reinf in snap_times:
                    df = self.extract_grammatical_information(threshold)
                    df.to_excel(f,sheet_name=str(self.n_reinf))

    
    def respond(self,stimuli_stream,s1,s2_index):
        # get the s2 stimuli and make it a chunk
        s2 = SChunk(stimuli_stream.stimuli[s2_index])

        response = self.choose_behaviour((s1,s2))

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
            else:
                self.border_before = stimuli_stream.border_before[s2_index]
                new_s1,s2_index = s2, s2_index + 1

            self.border_within = False
              
        else: # some type of chunking occurs
            # Check if there was a border
            if not self.border_within:
                self.border_within = stimuli_stream.border_before[s2_index]
            
            # Perform chunking at correct level
            new_s1 = s1.chunk_at_depth(s2,depth=s1.get_depth()+1-response) 
            s2_index+=1      
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
        #print(self.events)
        

        
        for couple,r in self.events:
            #substantial_couple= (str(couple[0]),str(couple[1]))
            #print('reinforcement')
            #Q = self.behaviour_repertoire[substantial_couple][r]
            subevents=[(couple,r)]
            subpairs = self.get_sub_couples(couple)
            for pair in subpairs:
                substantial_pair = (str(pair[0]),str(pair[1]))
                self.update_repertoire(pair)
                if r < len(self.behaviour_repertoire[substantial_pair]):
                    subevents.append((pair,r))
                    # Q += self.behaviour_repertoire[substantial_pair][r]
                    #print(subevents)
            for p,rr in subevents:
                substantial_p = (str(p[0]),str(p[1]))
                self.behaviour_repertoire[substantial_p][rr] += Learner.alpha * (u - self.behaviour_repertoire[substantial_p][rr])

        # Clear working memory
        self.events = []
        
    def extract_grammatical_information(self,threshold):
        grammar = list()
        for key,value in self.behaviour_repertoire.items():
            if max(value)>threshold:
                grammar.append((key[0],key[1],list(value).index(max(value)),max(value)))
        df = pd.DataFrame(grammar, columns = ['s1','s2','Index','Value'])
        return df


class RWLearner(Learner):
    
    def __init__(self, n_trials = 14, border = 'next'):
        super().__init__(n_trials =n_trials, border = border)
        
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
        weights = np.exp(RWLearner.beta * z)

        response = random.choices(options,weights/np.sum(weights))
        return response[0] 
        
    def reinforce(self, reinforcement = 'positive'):
        #print('call of reinforce')
        # for each events reinforce behaviour associated to chunk
        if reinforcement == 'positive':
            u = Learner.positive_reinforcement
        elif reinforcement == 'negative':
            u = Learner.negative_reinforcement
        #print(self.events)
        

        
        for couple,r in self.events:
            substantial_couple= (str(couple[0]),str(couple[1]))
            #print('reinforcement')
            Q = self.behaviour_repertoire[substantial_couple][r]
            subevents=[(couple,r)]
            subpairs = self.get_sub_couples(couple)
            for pair in subpairs:
                substantial_pair = (str(pair[0]),str(pair[1]))
                self.update_repertoire(pair)
                if r < len(self.behaviour_repertoire[substantial_pair]):
                    subevents.append((pair,r))
                    Q += self.behaviour_repertoire[substantial_pair][r]
                    #print(subevents)
            for p,rr in subevents:
                substantial_p = (str(p[0]),str(p[1]))
                self.behaviour_repertoire[substantial_p][rr] += RWLearner.alpha * (u - Q)

        # Clear working memory
        self.events = []    
    

# class TypeLearner(Learner):
#     initial_value_type = 0.
#     good_type_value = 2.
#     bad_type_value = -1.5
#     threshold_for_decision = 24

    
#     def __init__(self, n_trials = 14, border = 'next'):
#         super().__init__(n_trials =n_trials, border = border)
        

#         self.stimulus_types_dict = dict() #keys are strings represented a single stimulus and values are dictionary with types and values associated to them.
#         self.typatory = dict() # keys are types and values are related to how good the type is
#         self.typing_events = [] # encodes the current list of typings (format to be defined)

     
#     def add_chunk(self, chunk): # chunk must be a string
#         if chunk not in self.stimulus_types_dict:
#             self.stimulus_types_dict[chunk] = {}
            
#     def update_stimulus_type_association(self,s):
#         if type(s.structure) != list:
#             self.add_chunk(str(s))
                         
#     def get_sub_couples(self, couple):
#         sub_pairs = []
#         for s in couple[0].get_right_subchunks(couple[0].get_depth()):
#             self.update_stimulus_type_association(s)

#             sub_pairs.append((s,couple[1]))
#             self.update_repertoire((s,couple[1]))
#         return sub_pairs
    
#     def good_types(self,chunk): # chunk should be a string
#         if chunk in self.stimulus_types_dict:
#             good_types = dict()
#             for t,v in self.stimulus_types_dict[chunk].items():
#                 if v >= TypeLearner.good_type_value:
#                     good_types[t] = v
#             if len(good_types) == 0:
#                 return None
#             else:
#                 return good_types
#         else:
#             return None
        
    
#     def neutral_types(self,chunk): # chunk should be a string
#         if chunk in self.stimulus_types_dict:
#             good_types = dict()
#             for t,v in self.stimulus_types_dict[chunk].items():
#                 if v > TypeLearner.bad_type_value:
#                     good_types[t] = v
#             if len(good_types) == 0:
#                 return None
#             else:
#                 return good_types
#         else:
#             return None   
        
#     def neutral_start_types(self,chunk): # chunk should be a string
#         if chunk in self.stimulus_types_dict:
#             good_types = dict()
#             for t,v in self.stimulus_types_dict[chunk].items():
#                 if v > TypeLearner.bad_type_value and t.is_start():
#                     good_types[t] = v
#             if len(good_types) == 0:
#                 return None
#             else:
#                 return good_types
#         else:
#             return None  
        
#     def bad_types(self,chunk):
#         if chunk in self.stimulus_types_dict:
#             good_types = dict()
#             for t,v in self.stimulus_types_dict[chunk].items():
#                 if v <= TypeLearner.bad_type_value:
#                     good_types[t] = v
#             if len(good_types) == 0:
#                 return None
#             else:
#                 return good_types
#         else:
#             return None
        
#     def good_starting_types(self,chunk): # chunk should be a string
#         good_types = self.good_types(chunk)
#         if good_types != None:
#             good_start_types = dict()
#             for t,v in good_types.items():
#                 if t.is_start():
#                     good_start_types[t]=v
#             if len(good_start_types) == 0:
#                 return None
#             else:
#                 return good_start_types
#         else:
#             return None
            
        
#     def show_good_types(self):
#         for key, dic in self.stimulus_types_dict.items():
#             gt = self.good_types(key)
#             if gt != None:
#                 print(key)
#                 print(gt)
                
#     def show_bad_types(self):
#         for key, dic in self.stimulus_types_dict.items():
#             gt = self.bad_types(key)
#             if gt != None:
#                 print(key)
#                 print(gt)
                
#     def show_good_starting_types(self):
#         for key, dic in self.stimulus_types_dict.items():
#             gst = self.good_starting_types(key)
#             if gst != None:
#                 print(key)
#                 print(gst)
                
#     def associate_type_to_chunk(self,chunk,t): 
#         if t not in self.typatory:
#             self.typatory[t] = {str(chunk)}
#         else:
#             self.typatory[t].add(str(chunk))
#         if t not in self.stimulus_types_dict[str(chunk)]:
#             self.stimulus_types_dict[str(chunk)][t] = TypeLearner.initial_value_type
#             chunk.structure.retype(t.formula,TypeLearner.initial_value_type)
#         else:
#             chunk.structure.retype(t.formula,self.stimulus_types_dict[str(chunk)][t])
            

#     # should be ok
#     def merge_dicts(self,dic1,dic2):
#         merge_dict = dict()
#         for t,v in dic1.items():
#             merge_dict[(1,t)] = v
#         for t,v in dic2.items():
#             merge_dict[(2,t)] = v
#         return merge_dict
    
#     def choose_type_from_dict(self,dico):# maybe return the value as well, for retyping
#         keys = list(dico.keys())
#         values = np.array(list(dico.values()))
#         weights = np.exp(TypeLearner.beta * values)
#         response = random.choices(keys,weights/np.sum(weights))
#         #print(response)
#         return response[0]
    
#     def compatible_t2(self,t1,s2):
#         success = False
#         types = self.neutral_types(str(s2))
#         if types:
#             t2 = self.choose_type_from_dict(types)
#             while not success and len(types)!= 0:
#                 if t1.is_compatible(t2):
#                     success = True
#                     return success, t2
#                 if not success:
#                     del types[t2]
#                     if len(types)!=0:
#                         t2 = self.choose_type_from_dict(types)
#             return False, None
#         else:
#             return False,None
    
#     def compatible_t1(self,t2,s1):
#         success = False
#         types = self.neutral_start_types(str(s1))
#         if types:
#             t1 = self.choose_type_from_dict(types)
#             while not success and len(types)!= 0:
#                 if t1.is_compatible(t2):
#                     success = True
#                     return success, t1
#                 if not success:
#                     del types[t1]
#                     if len(types)!=0:
#                         t1 = self.choose_type_from_dict(types)
#             return False, None
#         else:
#             return False,None
    
#     def assign_t1(self,t2,s1,s2):
#         if t2.is_expecting_before():
#             t1 = Type(t2.left_type())
#             if self.bad_types(str(s1)):
#                 if t1 in self.bad_types(str(s1)):
#                     t1 = self.choose_primitive_type(s1)
#                     self.associate_type_to_chunk(s1, t1)
#                     # need to retype s2
#                     mod_type = Type(t1.formula+'u'+t2.left_type())
#                     self.associate_type_to_chunk(s2,mod_type+Type(s2.structure.type))
#                     #s2.structure.retype(mod_type+Type(s2.structure.type),self.stimulus_types_dict[mod_type+Type(s2.structure.type])
#                 else:
#                     self.associate_type_to_chunk(s1, t1)
#             else:
#                 if len(s1) == 1:
#                     self.associate_type_to_chunk(s1, t1)
#                 else:
#                     self.type_chunk(s1,t1)
#                 # test whether s1 is complex or not. If simple, associate type. If complex type_chunk
#             # try fullfilling expectation
#             # Otherwise retype both s1 and s2
#             pass
#         else:
#             if t2 == Type('0'):
#                 pass
#             else:
#                 # Need to test whether s1 is simple or complex
#                 t1 = self.choose_type_expecting_after(s1,t2)
#                 if len(s1)==1:
#                     self.associate_type_to_chunk(s1, t1)
#                 else:
#                     self.type_chunk(s1,t1)

#         # check whether t1 expect anything after if so fullfill expectation otherwise check if type is sentence, in that case assign new primitive, otherwise, create a type that reduces to a sentence.
#         #return t1
    
#     def assign_t2(self,t1,s1,s2): # need to pass s1 as well in case of retyping...
#         if t1.is_expecting_after():
#             t2 = Type(t1.right_type())
#             if self.bad_types(str(s2)):
#                 if t2 in self.bad_types(str(s2)):
#                     # t2 is bad for s2, so assign new primitive to s2 and retype s1 accordingly
#                     t2 = self.choose_primitive_type(s2)
#                     self.associate_type_to_chunk(s2, t2)
#                     new_stuff = s1.retype_from_expectation(Type(t1.right_type()),t2,self.stimulus_types_dict)
#                     print('new stuff')
#                     print(new_stuff)
#                     if new_stuff:
#                         self.associate_type_to_chunk(new_stuff[0], new_stuff[1])
#                         pass
#                     # here I need to check that whatever new type I create it has an entry in the stimulus_types_dict...
#                 else:
#                     # t2 is ok for s2, so assign it, do not touch s1
#                     self.associate_type_to_chunk(s2, t2)
#             else:
#                 # t2 is ok for s2 and there was no bad types for s2, type s2 and don't touch s1
#                 self.associate_type_to_chunk(s2, t2)
            
                    
#             #pass
#         else:
#             if t1 == Type('0'):
#                 pass
#             else:
#                 t2 = self.choose_type_expecting_before(s2,t1)
#                 self.associate_type_to_chunk(s2, t2)
#         # check whether t1 expect anything after if so fullfill expectation otherwise check if type is sentence, in that case assign new primitive, otherwise, create a type that reduces to a sentence.
#         #return t2
    
#     def choose_primitive_type(self,s): # s is a SChunk object
#         index = 0
#         if self.bad_types(str(s)):
#             while Type(str(index)) in self.bad_types(str(s)):
#                 index += 1
#         #self.associate_type_to_chunk(s, Type(str(index))) # Don't do that here, because s might be complex
#         # Maybe do the assignment here using the associate_type function
#         return Type(str(index))
    
#     def choose_type_expecting_after(self,s,t): # Choose a type that is not bad for s that expect t after
#         index = 0
#         if self.bad_types(str(s)):
#             while Type(str(index)+'o'+t.left_type()) in self.bad_types(str(s)):
#                 index += 1
#         #self.associate_type_to_chunk(s, Type(str(index))) # Don't do that here, because s might be complex
#         # Maybe do the assignment here using the associate_type function
#         return Type(str(index)+'o'+t.left_type())
    
#     def choose_type_expecting_before(self,s,t): # Choose a type that is not bad for s that expect t before
#         index = 0
#         if self.bad_types(str(s)):
#             while Type(t.right_type()+'u'+str(index)) in self.bad_types(str(s)):
#                 index += 1
#         #self.associate_type_to_chunk(s, Type(str(index))) # Don't do that here, because s might be complex
#         # Maybe do the assignment here using the associate_type function
#         return Type(t.right_type()+'u'+str(index))

#     def assign_types(self,s1,s2):
#         # print(type(s1.structure))
#         if s1.is_consistent(): # This means that s1 is typed and reduce to something.
#             print('s1 consistent')
#             types_s1 = s1.right_types_dict()
#             t1 = self.choose_type_from_dict(types_s1)
#             default = t1
#             success = False
#             while not success and len(types_s1)!= 0:
#                 success, t2 = self.compatible_t2(t1,s2)
#                 if not success:
#                     del types_s1[t1]
#                     if len(types_s1)!=0:
#                         t1 = self.choose_type_from_dict(types_s1)
#             if not success:
#                 t1 = default
#                 self.assign_t2(t1,s1,s2)
#             else:
#                 self.associate_type_to_chunk(s2, t2)
#         else:
#             print('s1 not consistent')
#             #print('s1 not typed')
#             #print(self.typed_structure)
#             #print(s1)
#             # NEEDS TO BE CHANGED: Jointly look through both typings using merge_dict...
#             good_starting_types = self.good_starting_types(str(s1))
#             good_types_s2 = self.good_types(str(s2))
            
            
            
#             #print(good_starting_types)
#             if good_starting_types != None and good_types_s2 == None:
#                 print('Case 1')
#                 #print('s1 get typed')
#                 success = False
#                 t1 = self.choose_type_from_dict(good_starting_types)
#                 default = t1
#                 # Look for compatible t2
#                 while not success and len(good_starting_types) !=0:
#                     success, t2 = self.compatible_t2(t1,s2)
#                     if not success:
#                         del good_starting_types[t1]
#                         if len(good_starting_types) != 0:
#                             t1 = self.choose_type_from_dict(good_starting_types)
#                 if not success:
#                     t1 = default
#                     self.assign_t2(t1,s1,s2)
#                     self.associate_type_to_chunk(s1, t1)
#                 else:
#                     self.associate_type_to_chunk(s1, t1)
#                     self.associate_type_to_chunk(s2, t2)
     
#             elif good_starting_types == None and good_types_s2 != None:
#                 print('Case 2')
#                 success = False
#                 t2 = self.choose_type_from_dict(good_types_s2)
#                 default = t2
#                 while not success and len(good_types_s2) !=0:
#                     success, t1 = self.compatible_t1(t2,s1)
#                     if not success:
#                         del good_types_s2[t2]
#                         if len(good_types_s2)!=0:
#                             t2 = self.choose_type_from_dict(good_types_s2)
#                 if not success:
#                     t2 = default
#                     self.associate_type_to_chunk(s2, t2)
#                     self.assign_t1(t2,s1,s2) # Inside this function, if s1 is complex one should use type_chunk
#                 else:
#                     self.associate_type_to_chunk(s1, t1)
#                     self.associate_type_to_chunk(s2, t2)
                
#                 # Need to check if s1 is complex use split otherwise update self.typed_structure
#                 #self.associate_type_to_chunk(s1, t1) # if s1 is complex should be done inside the assign function
#                 #self.associate_type_to_chunk(s2, t2)
            
#             elif good_starting_types != None and good_types_s2 != None:
#                 print('Case 3')
#                 merge = self.merge_dicts(good_starting_types, good_types_s2)
#                 success = False
#                 (index,t) = self.choose_type_from_dict(merge)
#                 default = (index,t)
#                 while not success and len(merge)!=0:
#                     if index ==1:
#                         # type 1 assign to s2 if compatible t2 for s2
#                         t1 = t
#                         success, t2 = self.compatible_t2(t1,s2)
#                     elif index == 2:
#                         t2 = t
#                         success, t1 = self.compatible_t1(t2,s1)
#                         # type 2 assign backwards to s1 if compatible t1 for s1
#                     else:
#                         print('Problem 1')
#                     if not success:
#                         del merge[(index,t)]
#                         if len(merge)!= 0:
#                             (index,t) = self.choose_type_from_dict(merge)
#                 if not success:
#                     if default[0]==1:
#                         # Default assignment
#                         t1 = default[1]
#                         self.assign_t2(t1,s1,s2)
#                         self.associate_type_to_chunk(s1, t1)
#                         pass
#                     elif default[0]==2:
#                         # Default assignment
#                         t2 = default[1]
#                         self.associate_type_to_chunk(s2, t2)
#                         self.assign_t1(t2,s1,s2) # Inside this function, if s1 is complex one should use type_chunk
#                         pass
#                     else:
#                         print('Problem 2')
#                 else:
#                     self.associate_type_to_chunk(s1, t1)
#                     self.associate_type_to_chunk(s2, t2)
#                     pass
                        
#                 #self.associate_type_to_chunk(s1, t1) # if s1 is complex should be done inside the assign function
#                 #self.associate_type_to_chunk(s2, t2)
                
#             else:   
#             #print(self.typed_structure)
#                 #print('No typing')
#                 #print(self.typed_structure)
#                 pass
    
#     def type_chunk(self,s1,type_to_split):
#         # Update typing event here
#         #self.typing_events.append((s1,type_to_split))
#         structure = s1.remove_structure()
#         types = dict()

#         # extract possible types by looping over the element of structure
#         for s in structure:
#             types[str(s)] = self.good_types(str(s))
        
#         # Check if any of the elements are typed
#         anytypes = False
#         for key,value in types.items():
#             if value != None:
#                 anytypes = True
#                 break
        
#         if not anytypes:
#             if type(s1.structure) is list:
#                 new_s1 = s1.get_s1()
#                 new_s2 = s1.get_s2()
#                 [t1,t2] = type_to_split.split()
#                 self.type_chunk(new_s1,t1)
#                 self.type_chunk(new_s2,t2)
#             else:
#                 self.associate_type_to_chunk(s1, type_to_split)
#                 #s1.structure.retype(str(type_to_split),TypeLearner.initial_value_type)
                
                
#         else:
#             # Check from bottom up if there is a possible typing
#             if type(s1.structure) is list:
#                 new_s1 = s1.get_s1()
#                 new_s2 = s1.get_s2()
#                 # Check if s1 and/or s2 are primitive
#                 if type(new_s1.structure) is not list and type(new_s2.structure) is not list:
#                     #print('both are primitive')
#                     # 1 typed, 2 typed, both typed?
#                     if types[str(new_s1.structure)]==None and types[str(new_s2.structure)]!=None:
#                         #print('Case 1')
#                         success = False
#                         t2 = self.choose_type_from_dict(types[str(new_s2.structure)])
#                         default = t2
#                         while not success and len(types[str(new_s2.structure)])!=0:
#                             success, t1 = self.choose_compatible_t1(t2,new_s1,type_to_split)
#                             if not success:
#                                 del types[str(new_s2.structure)][t2]
#                                 if len(types[str(new_s2.structure)])!=0:
#                                     t2 = self.choose_type_from_dict(types[str(new_s2.structure)])
#                         if not success:
#                             # POSSIBLY NEW TYPES
#                             [t1,t2] = type_to_split.split(pu=0,prim='New')
#                         self.associate_type_to_chunk(new_s1, t1)
#                         self.associate_type_to_chunk(new_s2, t2)
#                         #new_s1.structure.retype(str(t1),self.stimulus_types_dict[str(new_s1.structure)][t1])
#                         #new_s2.structure.retype(str(t2),self.stimulus_types_dict[str(new_s2.structure)][t2])
        
#                     elif types[str(new_s1.structure)]!=None and types[str(new_s2.structure)]==None:
#                         #print('Case 2')
#                         success = False
#                         t1 = self.choose_type_from_dict(types[str(new_s1.structure)])
#                         default = t1
#                         while not success and len(types[str(new_s1.structure)])!=0:
#                             success, t2 = self.choose_compatible_t2(t1,new_s2,type_to_split)
#                             if not success:
#                                 del types[str(new_s1.structure)][t1]
#                                 if len(types[str(new_s1.structure)])!=0:
#                                     t1 = self.choose_type_from_dict(types[str(new_s1.structure)])
#                         if not success:
#                             # POSSIBLY NEW TYPES
#                             [t1,t2] = type_to_split.split(pu=1,prim='New')
#                         self.associate_type_to_chunk(new_s1, t1)
#                         self.associate_type_to_chunk(new_s2, t2)
#                         #new_s1.structure.retype(str(t1),self.stimulus_types_dict[str(new_s1.structure)][t1])
#                         #new_s2.structure.retype(str(t2),self.stimulus_types_dict[str(new_s2.structure)][t2])
#                     else:
#                         #print('Case 3')
#                         success = False
#                         merge = self.merge_dicts(types[str(new_s1.structure)], types[str(new_s2.structure)])
#                         (index,t) = self.choose_type_from_dict(merge)
#                         default = (t,index)
#                         while not success and len(merge) != 0:
#                             if index == 1:
#                                 t1 = t
#                                 success, t2 = self.choose_compatible_t2(t1,new_s2,type_to_split)
#                                 if not success:
#                                     del merge[(index,t)]
#                                     if len(merge) != 0:
#                                         (index,t) = self.choose_type_from_dict(merge)
#                             elif index ==2:
#                                 t2 = t
#                                 success, t1 = self.choose_compatible_t1(t2,new_s1,type_to_split)
#                                 if not success:
#                                     del merge[(index,t)]
#                                     if len(merge) != 0:
#                                         (index,t) = self.choose_type_from_dict(merge)
#                             else:  
#                                 print('Problem here')
#                         if not success:
#                             # Possibly new types
#                             if default[1]==1:
#                                 t1 = default[0]
#                                 [t1,t2] = type_to_split.split(pu=1,prim='New',bad_s1=self.bad_types(str(new_s1)))
                                
#                             elif default[1]==2:
#                                 t2 = default[0]
#                                 [t1,t2] = type_to_split.split(pu=0,prim='New',bad_s2 = self.bad_types(str(new_s2)))
#                             else:
#                                 print('Problem here')
#                         self.associate_type_to_chunk(new_s1, t1)
#                         self.associate_type_to_chunk(new_s2, t2)  
#                         #new_s1.structure.retype(str(t1),self.stimulus_types_dict[str(new_s1.structure)][t1])
#                         #new_s2.structure.retype(str(t2),self.stimulus_types_dict[str(new_s2.structure)][t2])
#                         # Merge dict, choose dominant, loop through
#                         pass
#                 elif type(new_s1.structure) != list and type(new_s2.structure) == list:
#                     #print('s1 primitive, s2 complex')
#                     # where are the typings
#                     #print(types)
#                     #print(new_s1)
#                     if types[str(new_s1.structure)] != None:
#                         # success = False
#                         t1 = self.choose_type_from_dict(types[str(new_s1.structure)])
#                         [t1,t2] = type_to_split.split(pu=1,prim=t1)
#                         self.associate_type_to_chunk(new_s1, t1)
#                         #new_s1.structure.retype(str(t1),self.stimulus_types_dict[str(new_s1.structure)][t1])
#                         #new_s2.structure.retype(str(t2),self.stimulus_types_dict[str(new_s2.structure)][t2])
#                         self.type_chunk(new_s2,t2)
#                     else:
#                         [t1,t2] = type_to_split.split()
#                         self.associate_type_to_chunk(new_s1,t1)
#                         #new_s1.structure.retype(str(t1),self.stimulus_types_dict[str(new_s1.structure)][t1])
#                         #new_s2.structure.retype(str(t2),self.stimulus_types_dict[str(new_s2.structure)][t2])
#                         self.type_chunk(new_s2,t2)
#                 elif type(new_s2.structure) != list and type(new_s1.structure) == list:
#                     #print('s2 primitive, s1 complex')
#                     if types[str(new_s2.structure)] != None:
#                         # success = False
#                         #print('Case 1')
#                         t2 = self.choose_type_from_dict(types[str(new_s2.structure)])
#                         [t1,t2] = type_to_split.split(pu=0,prim=t2)
#                         #print(t1)
#                         #print(t2)
#                         self.associate_type_to_chunk(new_s2, t2)
#                         #print(self.typing_events)
#                         self.type_chunk(new_s1,t1)
#                         #new_s1.structure.retype(str(t1),self.stimulus_types_dict[str(new_s1.structure)][t1])
#                         #new_s2.structure.retype(str(t2),self.stimulus_types_dict[str(new_s2.structure)][t2])
#                     else:
#                         #print('Case 2')
#                         [t1,t2] = type_to_split.split()
#                         self.associate_type_to_chunk(new_s2,t2)
#                         self.type_chunk(new_s1,t1)
#                         #new_s1.structure.retype(str(t1),self.stimulus_types_dict[str(new_s1.structure)][t1])
#                         #new_s2.structure.retype(str(t2),self.stimulus_types_dict[str(new_s2.structure)][t2])
#                 else:
#                     #print('both are complex')
#                     [t1,t2] = type_to_split.split()
#                     self.type_chunk(new_s2,t2)
#                     self.type_chunk(new_s1,t1)
                    
#     def choose_compatible_t1(self,t2,s1,type_to_split):
#         success = True
#         bad_types= self.bad_types(str(s1))
#         if bad_types == None:
#             if t2.is_primitive():
#                 [t1,t2]=type_to_split.split(pu=0,prim=t2)
#                 success = True
#             else:
#                 if len(t2.left_compatible_chunks())==0:
#                     #print('not expecting before')
#                     success = False
#                     t1 = None
#                 else:
#                     #print('expecting before')
#                     t1 = Type(t2.formula[0])
#                     if t1.is_compatible(t2):
#                         if t1+t2 == type_to_split:
#                             success = True
#                         else:
#                             success = False
#                             t1 = None
#                     else:
#                         success = False
#                         t1 = None
#         else:
#             if t2.is_primitive():
#                 [t1,t2]=type_to_split.split(pu=0,prim=t2)
#                 if t1 in self.bad_types(str(s1)):
#                     success = False
#                     t1 = None
#                 else:
#                     success = True

#             else:
#                 if len(t2.left_compatible_chunks())==0:
#                     success = False
#                     t1 = None
#                 else:
#                     t1 = Type(t2.formula[0])
#                     if t1.is_compatible(t2) and t1 not in self.bad_types(str(s1)):
#                         if t1+t2 == type_to_split:
#                             success = True
#                         else:
#                             success = False
#                             t1 = None
#                     else:
#                         success = False
#                         t1 = None
#         return success, t1


#     def choose_compatible_t2(self,t1,s2,type_to_split):
#         success = True
#         bad_types= self.bad_types(str(s2))
#         if bad_types == None:
#             if t1.is_primitive():
#                 [t1,t2]=type_to_split.split(pu=1,prim=t1)
#                 success = True
#             else:
#                 if len(t1.right_compatible_chunks())==0:
#                     #print('not expecting before')
#                     success = False
#                     t2 = None
#                 else:
#                     #print('expecting before')
#                     t2 = Type(t1.right_type())#Type(t1.formula[-1])
#                     if t1.is_compatible(t2):
#                         if t1+t2 == type_to_split:
#                             success = True
#                         else:
#                             success = False
#                             t1 = None
#                     else:
#                         success = False
#                         t1 = None
#         else:
#             if t1.is_primitive():
#                 [t1,t2]=type_to_split.split(pu=1,prim=t1)
#                 if t1 in self.bad_types(str(s2)):
#                     success = False
#                     t1 = None
#                 else:
#                     success = True

#             else:
#                 if len(t1.right_compatible_chunks())==0:
#                     success = False
#                     t2 = None
#                 else:
#                     t2 = Type(t1.right_type())
#                     if t1.is_compatible(t2) and t2 not in self.bad_types(str(s2)):
#                         if t1+t2 == type_to_split:
#                             success = True
#                         else:
#                             success = False
#                             t2 = None
#                     else:
#                         success = False
#                         t2 = None
#         return success, t2 

#     def learn(self,stimuli_stream):
#         # initialize stimuli
#         s1 = SChunk(Stimuli(stimuli_stream.stimuli[0]))

#         self.update_stimulus_type_association(s1)
#         s2_index = 1
#         #for t in range(self.n_trials):
#         while self.n_reinf <= self.n_trials:
#             s1, s2_index = self.respond(stimuli_stream, s1, s2_index)

            
    
#     def respond(self,stimuli_stream,s1,s2_index):
#         # get the s2 stimuli and make it a chunk
#         s2 = SChunk(Stimuli(stimuli_stream.stimuli[s2_index]))

#         # update chunkatory
#         self.update_stimulus_type_association(s2)
        
        
#         self.assign_types(s1,s2)
#         print('-------------')
#         print(s1)
#         print(s1.get_list_types())
#         print(s2)
#         print(s2.get_list_types())
        
#         #################################
#         # assignment of types
#         #################################
        
#         response = self.choose_behaviour((s1,s2)) # Should be modified to take into account types in the decision
#         print('response')
#         print(response)

#         self.events.append(((s1,s2),response))
        
                
#         if response == 0: # boundary placement
#             #print('Border')
#             # increment the number of reinforcement events by 1
#             self.n_reinf += 1 
#             # check if border is correctly placed
#             is_border = stimuli_stream.border_before[s2_index]

#             if is_border and not self.border_within and self.border_before:
#                 #print('Good Unit')
#                 # perform positive reinforcement
#                 # Store sentence (not cognitively plausible but used for grammar extraction)
#                 self.sentences.add(str(s1))
                
#                 #################################
#                 # type sentence if new comes here
#                 #################################
#                 if not s1.is_consistent():
#                     self.type_chunk(s1, Type('0'))
#                     self.typing_events= s1.get_typing_events()
                    
#                     self.reinforce_typings()
#                     # reinforce typings here
#                 else:
#                     if s1.reduce() == Type('0'):
#                         self.typing_events= s1.get_typing_events()
#                         self.reinforce_typings(reinforcement = 'positive')
#                     else:
#                         self.typing_events= s1.get_typing_events()
#                         self.reinforce_typings(reinforcement = 'negative')
#                     # reinforce positively s1 typings if they reduce to a sentence, negatively otherwise and negatively s2 typings waiting something before

#                 # Perform reinforcement
#                 self.reinforce(reinforcement = 'positive')  
                
#                 #################################
#                 # reinforcement of types
#                 #################################
                
#                 # self.reinforce_typings() To be defined                  
#                 # update the success list
#                 self.success.append(1)
#                 # for postprocessing, store length of the sentence
#                 self.sent_len.append(stimuli_stream.length_current_sent(s2_index - 1))
#             else:
#                 #print('Bad Unit')
#                 # perform negative reinforcement
#                 self.reinforce(reinforcement = 'negative')
#                 # update the success list
                
#                 #################################
#                 # reinforcement of types
#                 #################################
#                 if s1.is_consistent():
#                     if s1.reduce() == Type('0'):
#                         self.typing_events= s1.get_typing_events()
#                         self.reinforce_typings(reinforcement='negative')
#                     # reinforce negatively s1 typings
#                 else:
#                     # don't do anything with typings
#                     pass

#                 self.success.append(0)
#                 self.sent_len.append(stimuli_stream.length_current_sent(s2_index))
#             # Next beginning of sentence becomes
            
#             if self.border_type == 'next':
#                 new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
#                 new_s1 = SChunk(Stimuli(new_s1))

#             else:
#                 self.border_before = stimuli_stream.border_before[s2_index]
#                 new_s1,s2_index = s2, s2_index + 1

                
#             #self.chunks.add(new_s1)
#             self.update_stimulus_type_association(new_s1)

#             self.border_within = False
            
              
#         else: # some type of chunking occurs
#             # Check if there was a border
#             if not self.border_within:
#                 self.border_within = stimuli_stream.border_before[s2_index]
            
#             # Perform chunking at correct level
#             new_s1 = s1.chunk_at_depth(s2,depth=s1.get_depth()+1-response) 
#             s2_index+=1
#             self.update_stimulus_type_association(new_s1)

            
#         return new_s1, s2_index
    
#     def choose_behaviour(self,couple):
#         substantial_couple = (str(couple[0]),str(couple[1]))
#         self.update_repertoire(couple)
#         b_range = len(self.behaviour_repertoire[substantial_couple])
#         options = [i for i in range(b_range)]
#         z = deepcopy(self.behaviour_repertoire[substantial_couple])
#         z_type = [0 for i in range(len(z))]

#         subpairs = self.get_sub_couples(couple)
        
#         norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
#         # Accumulate support from subchunks
#         for pair in subpairs:
#             substantial_pair = (str(pair[0]),str(pair[1]))
#             lenp = len(self.behaviour_repertoire[substantial_pair])
#             z[:lenp] += self.behaviour_repertoire[substantial_pair]
#             # Take the average
#         z /= norm_vec
        
#         if couple[0].is_consistent() and couple[1].is_consistent():
#             if couple[0].reduce() == Type('0') and Type(couple[1].structure.type).is_expecting_before() and Type(couple[1].structure.type).left_type() == '0':
#                 print('support for chunking at highest level')
#                 print(couple[0].right_types())
#                 print(couple[1].reduce())
#                 ind = 0
#                 new_index = None
#                 for t in couple[0].right_types():
#                     if t.is_compatible(couple[1].reduce()):
#                         new_index = ind
#                     ind += 1
#                 print('index: '+str(new_index))
#                 print('response: '+ str(-(new_index +1)))
#                 z_type[-(new_index +1)] = (couple[0].right_values()[new_index] + couple[1].average())/2
#                 value = z_type[-(new_index +1)]
                
#             elif couple[0].reduce() == Type('0') and not Type(couple[1].structure.type).is_expecting_before():
#                 print('support for border')
#                 z_type[0] = (couple[0].average() + couple[1].average())/2
#                 value = (couple[0].average() + couple[1].average())/2
#             else:
#                 print('support for chunking level to be defined')
#                 print(couple[0].right_types())
#                 print(couple[1].reduce())
#                 ind = 0
#                 new_index = None
#                 for t in couple[0].right_types():
#                     if t.is_compatible(couple[1].reduce()):
#                         new_index = ind
#                     ind += 1
#                 print('index: '+str(new_index))
#                 print('response: '+ str(-(new_index +1)))
#                 z_type[-(new_index +1)] = (couple[0].right_values()[new_index] + couple[1].average())/2
#                 value = z_type[-(new_index +1)]
#         #################################
#         # Support from assigned types comes here
#         #################################
#             if value >= TypeLearner.threshold_for_decision:
#                 z = z + z_type
        
#         weights = np.exp(Learner.beta * z)

#         response = random.choices(options,weights/np.sum(weights))
#         return response[0]  
    


#     def reinforce_typings(self,reinforcement = 'positive'):
#         if reinforcement == 'positive':
#             u = TypeLearner.positive_reinforcement
#         elif reinforcement == 'negative':
#             u = TypeLearner.negative_reinforcement
            
#         for s,t in self.typing_events:
#             if t != Type(''):
#                 self.stimulus_types_dict[s][t] += TypeLearner.alpha * (u - self.stimulus_types_dict[s][t]) 
            

    
