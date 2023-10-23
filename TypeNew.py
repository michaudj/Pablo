# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:21:27 2023

@author: jmd01
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:22:29 2023

@author: jmd01
"""

import re
import json
from json import JSONEncoder
import copy
from itertools import accumulate
from random import random


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
            
        if random() < pu:
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




class TChunk():
    
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
        #self.type_dic = {}
        self.depth = self.get_depth()
        
    def __repr__(self):
        return str(self.structure)
    
    def __hash__(self):
        return hash(frozenset((self.structure)))
    
    def get_s1(self):
        [s1,s2] = self.structure
        s1 = TChunk(s1)
        s2 = TChunk(s2)
        return s1
    
    def get_s2(self):
        [s1,s2] = self.structure
        s1 = TChunk(s1)
        s2 = TChunk(s2)
        return s2
    
    def get_right_subchunks(self, depth):
        right_subchunks = []
        nested_list = self.structure[:]
        for d in range(depth):
            nested_list = nested_list[-1]
            right_subchunks.append(TChunk(nested_list))
        return right_subchunks
    
    def chunk_at_depth(self, other, depth=0):
        if type(self.structure)== Type:
            nested_list = self.structure
        else:
            nested_list = self.structure[:]
        
        if depth == 0:
            return TChunk([nested_list,other.structure])
        else:
            modify_element_at_depth(nested_list, depth, other.structure)
            return TChunk(nested_list)
    
    def get_depth(self):
        st = str(self.structure)
        match = re.search("]*$",st)
        return len(match.group(0))
    
    def remove_structure(self):
        if type(self.structure) is Type:
            return str(self.structure)
        else:
            return flatten(self.structure)
        
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
                    return s1.reduce().is_compatible(s2.structure)
                else:
                    return False
            elif type(s1.structure) == Type and type(s2.structure)!= Type:
                if s2.is_consistent():
                    return s1.structure.is_compatible(s2.reduce())
                else:
                    return False
            else:
                if s1.is_consistent() and s2.is_consistent():
                    return s1.reduce().is_compatible(s2.reduce())
                else:
                    return False
    
    
    def right_types(self):
        # Only works if TChunk is consistent!!!
        list_of_reduced_types = [self.reduce()]
        
        for chunk in self.get_right_subchunks(self.depth):
            list_of_reduced_types.append(chunk.reduce())
        return list_of_reduced_types
        

          
###############################################################################
#
#           Tests
#
###############################################################################
     
a = Type(r"aufubocodd")
print(a)
print(a.formula)
print(re.findall(r"o"+a.right_type()+"$",a.formula))
print(re.findall(r"^"+a.left_type()+"u",a.formula))
#print(a.right_compatible_chunks2())
#print(a.left_compatible_chunks2())

b = Type(r"aufubocodd")
print(b.left_type())
print(b.right_type())
print(b.get_primitives())


print('-------------------------------')
d = Type(r"ddoe")
e = Type(r"dd")
f = Type(r"a")
g = Type(r"auf")
h = Type(r"buauf")
#print(f.right_compatible_chunks2())
#print(f.left_compatible_chunks2())
print(f.right_compatible_chunks())
print(f.left_compatible_chunks())
print(b.is_primitive())
print(d.is_primitive())
print(e.is_primitive())
print(f.is_primitive())
print(h.is_primitive())
print(h.is_compatible(g))
print('Compatibility check for ghost types')

print(str(a)+'+'+str(d)+'='+str(a+d))
print(str(a)+'+'+str(e)+'='+str(a+e))
print(str(f)+'+'+str(a)+'='+str(f+a))
#print(str(g)+'+'+str(a)+'='+str(g+a))
#print(str(h)+'+'+str(a)+'='+str(h+a))




    
def reduce_types(types):
    # make a copy of the list of types
    remaining_types = types[:]
    
    # keep trying to reduce the list of types until it contains only one type
    while len(remaining_types) > 1:
        #print(remaining_types)
        # set the reduced flag to False
        reduced = False
        
        # iterate over the remaining types
        for i, type1 in enumerate(remaining_types[:-1]):
            type2 = remaining_types[i+1]
            # if the two types are compatible, reduce them and update the reduced flag
            if type1.is_compatible(type2):
                #print('add')
                #print(type1)
                #print(type2)
                #print('result')
                #print(type1+type2)
                remaining_types[i] = type1 + type2
                del remaining_types[i+1]
                reduced = True
                break
        
        # if the list of types was not reduced, raise a TypeError
        if not reduced:
            return remaining_types
    
    # return the remaining type
    return remaining_types

t1 = Type("1o3")
t3 = Type("3")
t2 = Type("1u0o2")
t5 = Type("dd")
t4 = Type("4u2o3")
ttt = Type('')
print(t1.is_primitive())
print(t2.is_primitive())
print(t3.is_primitive())
print(t4.is_primitive())
print(t5.is_primitive())
print(ttt.is_primitive())
print(ttt.is_empty())

print('???????????????????????????????')

tt = Type("0")
types = tt.split(pu=0.5)
print(types)
ttypes= types[0].split(pu=0.5,prim='New') + types[1].split(pu=0.5)
print(ttypes)
tctypes = []
for ttt in ttypes:
    tctypes.append(TChunk(ttt))
    
tttc =tctypes[0].chunk_at_depth(tctypes[1])
tttc2 = tctypes[2].chunk_at_depth(tctypes[3])
tchunk = tttc.chunk_at_depth(tttc2)
print(tchunk)
reduce_types(ttypes)
print('Reduced?')
print(Type.reduce(ttypes))
print(Type.is_sentence(ttypes))
#print(types[0].split())
#print(types[1].split())

#result = reduce_types([t1,t3, t2,t5, t4,t3])
#print(result)  # prints "a\\b\\c\\d"

tc =TChunk(b)
print(tc)
tcc = TChunk(t5)
print(tcc)
new_tc =TChunk([tc.structure,tcc.structure])
print(new_tc)
if new_tc.is_consistent():
    print(new_tc.reduce())
else:
    print('incompatible types')
#print(new_tc.is_consistent())
print('---------------------')

print(tchunk)
print('right types')
print(tchunk)
print(tchunk.right_types())
print('Test remove structure')
print(tchunk.remove_structure())
print(tchunk.get_right_subchunks(tchunk.depth))
#print(reduce_types(tchunk.remove_structure()))
list_of_reduced_types = reduce_types(tchunk.remove_structure())
for chunk in tchunk.get_right_subchunks(tchunk.depth):
    if type(chunk.structure) is not Type:
        #print(chunk.structure[0]+chunk.structure[1])
        list_of_reduced_types.append(chunk.structure[0]+chunk.structure[1])
    else:
        list_of_reduced_types.append(chunk.structure)
        #print(chunk.structure)
list_of_reduced_types.reverse()
print(list_of_reduced_types)


print('===================================')
tt = Type("0")
types = tt.split(pu=0.5)
print(types)
bad_s1 = {Type('0'):-1,Type('1'):-2}
bad_s2 = {Type('0'):-1,Type('1'):-2,Type('2'):-2}
ttypes= types[0].split(pu=0.5,prim='New',bad_s1=bad_s1,bad_s2=bad_s2) + types[1].split(pu=0.5)
print(ttypes)
tctypes = []
for ttt in ttypes:
    tctypes.append(TChunk(ttt))
    
tttc =tctypes[0].chunk_at_depth(tctypes[1])
tttc2 = tctypes[2].chunk_at_depth(tctypes[3])
tchunk = tttc.chunk_at_depth(tttc2)
print(ttypes)
reduce_types(ttypes)
print('Reduced?')
print(Type.reduce(ttypes))
print(Type.is_sentence(ttypes))

print('===================================')
print(tchunk)
print(tchunk.get_right_subchunks(tchunk.depth))
#print(tchunk.is_consistent())
#print(tchunk)
#print(tchunk.reduce())
#print(tchunk)
print('right types version 1')
print(tchunk.right_types())

#TChunk(2)
