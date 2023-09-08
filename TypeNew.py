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
from itertools import accumulate
from random import random

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
    
    def split(self,pu=0.5,prim=False): # should return two types that combine into the initial type
        if not prim:
            if random() < pu:
                return [self, Type(self.formula+"u"+self.formula )]
            else:
                return [Type(self.formula+"o"+self.formula ), self]
        else:
            prim_type = Type.create_primitive_type()
            if random() < pu:
                return [prim_type, Type(prim_type.formula+"u"+self.formula )]
            else:
                return [Type(self.formula+"o"+prim_type.formula ), prim_type]
        pass
    
    def is_start(self): # Checks whether the type is expecting something on the left. 
        return len(self.left_compatible_chunks()) == 0
    
    
    def left_compatible_chunks(self):
        substrings = re.findall(r".*?u", self.formula)
        substrings = list(accumulate(substrings))
        substrings = [re.sub(r"u$", "$", x) for x in substrings]
        substrings1 = [re.sub(r"^", r"^", x) for x in substrings]
        substrings2 = [re.sub(r"^", r"u", x) for x in substrings]
        return substrings1 + substrings2
    
    def right_compatible_chunks(self):
        substrings = re.findall(r".*?o", self.formula[::-1])
        substrings = list(accumulate(substrings))
        substrings = [re.sub(r"o$", "", x) for x in substrings]
        substrings = [x[::-1] for x in substrings]
        substrings1 = [re.sub(r"^", r"^", x) for x in substrings]
        substrings2 = [re.sub(r"$", r"o", x) for x in substrings1]
        substrings1 = [re.sub(r"$", r"$", x) for x in substrings1]
        return substrings1 + substrings2
    
    def is_primitive(self):
        return (len(self.right_compatible_chunks()) + len(self.left_compatible_chunks())) == 0
    
    
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
            
###############################################################################
#
#           Tests
#
###############################################################################
     
a = Type(r"aufubocod")
# print(a)
print(a.left_compatible_chunks())
# print(a.right_compatible_chunks())

b = Type(r"aufubocod")
d = Type(r"doe")
e = Type(r"d")
f = Type(r"a")
g = Type(r"auf")
h = Type(r"buauf")
print(b.is_primitive())
print(d.is_primitive())
print(e.is_primitive())
print(f.is_primitive())
print(h.is_primitive())

print(str(a)+'+'+str(d)+'='+str(a+d))
print(str(a)+'+'+str(e)+'='+str(a+e))
print(str(f)+'+'+str(a)+'='+str(f+a))
print(str(g)+'+'+str(a)+'='+str(g+a))
print(str(h)+'+'+str(a)+'='+str(h+a))


    
def reduce_types(types):
    # make a copy of the list of types
    remaining_types = types[:]
    
    # keep trying to reduce the list of types until it contains only one type
    while len(remaining_types) > 1:
        print(remaining_types)
        # set the reduced flag to False
        reduced = False
        
        # iterate over the remaining types
        for i, type1 in enumerate(remaining_types[:-1]):
            type2 = remaining_types[i+1]
            # if the two types are compatible, reduce them and update the reduced flag
            if type1.is_compatible(type2):
                print('add')
                print(type1)
                print(type2)
                print('result')
                print(type1+type2)
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
t5 = Type("4")
t4 = Type("4u2o3")
print(t1.is_primitive())
print(t2.is_primitive())
print(t3.is_primitive())
print(t4.is_primitive())
print(t5.is_primitive())

tt = Type("0")
types = tt.split(prim=True)
print(types)
ttypes= types[0].split(prim=True) + types[1].split()
print(ttypes)
reduce_types(ttypes)
#print(types[0].split())
#print(types[1].split())

#result = reduce_types([t1,t3, t2,t5, t4,t3])
#print(result)  # prints "a\\b\\c\\d"