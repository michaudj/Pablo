# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:49:31 2019

@author: jermi792
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import random
import numpy as np
random.seed()

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


class ProbabilisticGrammar:
    def __init__(self, terminals, non_terminals, production_rules,weights):
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.production_rules = production_rules
        self.weights = weights
        
    def __repr__(self):
        non_terminals_str = ', '.join(self.non_terminals)
        production_rules_str = '\n'.join([f"{k} -> {v}" for k, v in self.production_rules.items()])
        weights_str = '\n'.join([f"{k} -> {v}" for k, v in self.weights.items()])
        return f"terminals={self.terminals},\nnon_terminals=[{non_terminals_str}],\nproduction_rules=\n{production_rules_str},\nweights=\n{weights_str}"

    def __add__(self,other):
        new_terminals = list(set(self.terminals + other.terminals))
        new_non_terminals = list(set(self.non_terminals + other.non_terminals))

        new_production_rules = dict()
        new_production_rules.update(self.production_rules)
        new_production_rules.update(other.production_rules)

        new_weights = dict()
        new_weights.update(self.weights)
        new_weights.update(other.weights)

        return ProbabilisticGrammar(new_terminals, new_non_terminals, new_production_rules, new_weights)
    
    def generate_sentence(self, symbol):
        if symbol in self.terminals:
            return [symbol]
        
        rules = self.production_rules[symbol]
        weights = self.weights[symbol]
        
        chosen_rule = random.choices(rules, weights=weights)[0]
        
        return flatten([self.generate_sentence(s) for s in chosen_rule])
    
@dataclass
class RawInput:
    n_sentences: int 
    grammar: 'ProbabilisticGrammar' 
    
    stimuli: List[str] = field(init=False, default_factory=list)
    border_before: List[bool] = field(init=False, default_factory=list)
    sentences: List[List[str]] = field(init=False, default_factory=list)
    
    def __post_init__(self):
        self.sentences = [self.grammar.generate_sentence('S') for _ in range(self.n_sentences)]
        for sentence in self.sentences:
            self.stimuli.extend(sentence)
            self.border_before.extend([True] + [False] * (len(sentence) - 1))
    
    def next_beginning_sent(self, index: int) -> Tuple[str, int] | None:
        for i in range(index, len(self.border_before)):
            if self.border_before[i]:
                return (self.stimuli[i], i + 1)
        return None
    
    def length_current_sent(self, index: int) -> int:
        start = index
        while start > 0 and not self.border_before[start]:
            start -= 1 
        end = index + 1
        while end < len(self.border_before) and not self.border_before[end]:
            end +=1 
            
        return end - start
    
    @property 
    def number_of_sentences(self) -> int:
        return len(self.sentences)
    
    @property 
    def number_of_words(self) -> int:
        return len(self.stimuli)
    
    def __repr__(self) -> str:
        return f"<RawInput with {self.number_of_sentences} sentences and {self.number_of_words} words>"
    
    
@dataclass
class RawInputLazy:
    n_sentences: int 
    grammar: 'ProbabilisticGrammar' 
    
    stimuli: List[str] = field(init=False, default_factory=list)
    border_before: List[bool] = field(init=False, default_factory=list)
    #sentences: List[List[str]] = field(init=False, default_factory=list)
    
    def __post_init__(self):
        pass
        # self.sentences = [self.grammar.generate_sentence('S') for _ in range(self.n_sentences)]
        # for sentence in self.sentences:
        #     self.stimuli.extend(sentence)
        #     self.border_before.extend([True] + [False] * (len(sentence) - 1))
        
    def sentence_generator(self) -> Iterator[List[str]]:
        """Yields one generated sentence at a time."""
        for _ in range(self.n_sentences):
            yield self.grammar.generate_sentence('S')
            
    def fill_until(self, target_index: int):
        """Generate sentenes until stimuli is long enough to reach target_index."""
        gen = self.sentence_generator()
        try:
            while len(self.stimuli) <= target_index:
                sentence = next(gen)
                self.stimuli.extend(sentence)
                self.border_before.extend([True] + [False] * (len(sentence) - 1))
        except StopIteration:
            pass
    
    def next_beginning_sent(self, index: int) -> Tuple[str, int] | None:
        self.fill_until(index)
        for i in range(index, len(self.border_before)):
            if self.border_before[i]:
                return (self.stimuli[i], i + 1)
        return None
    
    def read_stimulus(self,index):
        self.fill_until(index)
        return self.stimuli[index]
    
    def length_current_sent(self, index: int) -> int:
        self.fill_until(index)
        start = index
        while start > 0 and not self.border_before[start]:
            start -= 1 
        end = index + 1
        while end < len(self.border_before) and not self.border_before[end]:
            end +=1 
            
        return end - start
    
    @property 
    def number_of_words(self) -> int:
        return len(self.stimuli)
    
    def __repr__(self) -> str:
        return f"<RawInputLazy with {self.number_of_words} words>"   



print('Defining the grammar')

# definition of the grammar
# Vocabulary
number_of_verbs = 5
number_of_nouns = 5
number_of_adj = 0
number_of_relpron = 2
number_of_det = 0

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]

terminals2 = flatten([verbs,nouns,adjs,relpron,det])
non_terminals2 = ['S', 'N','NP','VP','V','RelCl']

###############################################################
#
#       NVN language
#
###############################################################

# Grammatical rules
production_rulesNVN = {
    'S': [['N', 'VP','N']],
    'VP': [['V']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsNVN = {
    'S': [1],
    'VP': [1],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

cfgNVN = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVN)

# Grammatical rules
production_rulesRCP = {
    'S': [['N', 'VP'],['N','VP','RelCl'],['N','VP','RelCl','RelCl']],
    'VP': [['V','N'],['V']],
    'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsRCP = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [.5, .5],
    'RelCl': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

relweight = np.array([1/2**i for i in range(len(relpron))])
relweight /= np.sum(relweight)

nweight = np.array([1/2**i for i in range(len(nouns))])
nweight /= np.sum(nweight)

vweight = np.array([1/2**i for i in range(len(verbs))])
vweight /= np.sum(vweight)

weightsRCPZipf = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [.5, .5],
    'RelCl': relweight,
    'N': nweight,
    'V': vweight   
    }

# Context free grammar
cfgRCP = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesRCP,weightsRCPZipf)

raw = RawInputLazy(n_sentences=10, grammar=cfgRCP)

print(raw)