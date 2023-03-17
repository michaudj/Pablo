# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:22:29 2023

@author: jmd01
"""

class Type:
    # create a cache to store instances of the Type class
    cache = {}
    
    # override the __new__ method to check the cache for an existing instance
    def __new__(cls, name, left=None, right=None):
        # create a key for the instance based on its name, left, and right values
        key = (name, left, right)
        
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
    
    # initialize the instance with its name, left, and right values
    def __init__(self, name, left=None, right=None):
        self.left = left
        self.name = name
        self.right = right
        
    def __hash__(self):
        return hash(frozenset([self.left,self.name,self.right]))
        
    # define the string representation of the instance
    def __repr__(self):
        if self.left ==None and self.right == None:
            return f"{self.name}"
        elif self.left ==None:
            return f"{self.name}/{self.right}"
        elif self.right ==None:
            return f"{self.left}\\{self.name}"
        else:
            return f"{self.left}\\{self.name}/{self.right}"
        
    def __add__(self, other):
        if self.is_compatible(other):
            if self.right == other.name and other.left == None:
                return Type(self.name,self.left,  other.right)
            elif self.right == None and self.name == other.left:
                return Type(other.name,self.left,  other.right)
        else:
            raise TypeError("Incompatible types")
        
    def __eq__(self, other):
        return self is other
            
    def is_compatible(self, other):
        return (self.right == other.name and other.left == None) or (self.right == None and self.name == other.left)

# create some types
a = Type(0,left=None,  right=1)
b = Type(1,left=None,  right=2)
c = Type(2,left=0,  right=1)
d = Type(2,left=a,  right=b)
print(d)

# check if the types are


# check if the types are compatible
print(a.is_compatible(b))
print(b.is_compatible(c))
print(a.is_compatible(c))

# combine the types using the + operator
if a.is_compatible(b):
    result1 = a + b
    result2 = a + b
    print(result1)
    print(result1 == result2)
else:
    print("Incompatible types")

if c.is_compatible(b):
    result = c + b
    print(result)
else:
    print("Incompatible types")

if a.is_compatible(c):
    result = a + c
    print(result)
else:
    print("Incompatible types")
    
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
                remaining_types[i] = type1 + type2
                del remaining_types[i+1]
                reduced = True
                break
        
        # if the list of types was not reduced, raise a TypeError
        if not reduced:
            raise TypeError("Cannot reduce types")
    
    # return the remaining type
    return remaining_types[0]


t1 = Type(1,right=3)
t3 = Type(3)
t2 = Type(0,left=1,  right=2)
t5 = Type(4)
t4 = Type(2,left=4,  right=3)

result = reduce_types([t1,t3, t2,t5, t4,t3])
print(result)  # prints "a\\b\\c\\d"