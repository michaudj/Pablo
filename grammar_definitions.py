# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:28:09 2025

@author: jemi6917
"""

from RawInput import ProbabilisticGrammar
import numpy as np


def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def NVN(
    number_of_verbs = 2,
    number_of_nouns = 50
    ):
    
    verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]

    
    terminals2 = flatten([verbs,nouns])
    non_terminals2 = ['S', 'N','VP','V']
    
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
    
    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    vweight = np.array([1/(i+1) for i in range(len(verbs))])
    vweight /= np.sum(vweight)

    weightsNVNZipf = {
        'S': [1 ],
        'VP': [1],
        'N': nweight,
        'V': vweight   
        }
    
    return ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVNZipf)

def NVNadj(
    number_of_verbs = 20,
    number_of_nouns = 50,
    number_of_adjectives = 10
    ):
    
    verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
    adjectives = ['a' + str(i) for i in range(1, number_of_adjectives+1)]
    
    terminals2 = flatten([verbs,nouns,adjectives])
    non_terminals2 = ['S', 'N','VP','V', 'NP','A']
    
    # Grammatical rules
    production_rulesNVN = {
        'S': [['NP', 'VP','NP']],
        'VP': [['V']], 
        'NP': [['N'],['N','A'],['N','A','A']],
        'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
        'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
        'A': [['a' + str(i)] for i in range(1, number_of_adjectives+1)]
    }
    
    weightsNVN = {
        'S': [1],
        'VP': [1],
        'NP': [.7,.25,.05],
        'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
        'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],   
        'A': [1/number_of_adjectives for i in range(1, number_of_adjectives+1)]
        }
    
    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    vweight = np.array([1/(i+1) for i in range(len(verbs))])
    vweight /= np.sum(vweight)
    
    aweight = np.array([1/(i+1) for i in range(len(adjectives))])
    aweight /= np.sum(aweight)

    weightsNVNZipf = {
        'S': [1 ],
        'VP': [1],
        'NP': [.7,.25,.05],
        'N': nweight,
        'V': vweight,
        'A': aweight
        }
    
    return ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVNZipf)


def NVN_test(
    number_of_verbs = 20,
    number_of_nouns = 50):
    
    verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(40, number_of_nouns+40)]

    
    terminals2 = flatten([verbs,nouns])
    non_terminals2 = ['S', 'N','VP','V']
    
    # Grammatical rules
    production_rulesNVN = {
        'S': [['N', 'VP','N']],
        'VP': [['V']],
        'N': [['n' + str(i)] for i in range(40, number_of_nouns+40)],
        'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
    }
    
    weightsNVN = {
        'S': [1],
        'VP': [1],
        'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
        'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
        }
    
    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    vweight = np.array([1/(i+1) for i in range(len(verbs))])
    vweight /= np.sum(vweight)

    weightsNVNZipf = {
        'S': [1 ],
        'VP': [1],
        'N': nweight,
        'V': vweight   
        }
    
    return ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVNZipf)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgNVN)

def create_stimuliVNN():
    number_of_verbs = 20
    number_of_nouns = 50
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
    
    # Grammatical rules
    production_rulesNVN = {
        'S': [['VP','N','N']],
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
    
    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    vweight = np.array([1/(i+1) for i in range(len(verbs))])
    vweight /= np.sum(vweight)

    weightsNVNZipf = {
        'S': [1 ],
        'VP': [1],
        'N': nweight,
        'V': vweight   
        }
    
    return ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVNZipf)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgNVN)

def create_stimuliNNV():
    number_of_verbs = 10
    number_of_nouns = 60
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
    
    # Grammatical rules
    production_rulesNVN = {
        'S': [['N','N', 'VP']],
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
    

    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    vweight = np.array([1/(i+1) for i in range(len(verbs))])
    vweight /= np.sum(vweight)


    weightsNVNZipf = {
        'S': [1 ],
        'VP': [1],
        'N': nweight,
        'V': vweight   
        }
    
    return ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVNZipf)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgNVN)


def create_stimuliRCP():
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


    # Grammatical rules
    production_rulesRCP = {
        'S': [['N', 'VP'],['N','VP','RelCl'],['N','VP','RelCl','RelCl']],
        'VP': [['V','N'],['V']],
        'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
        'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
        'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
    }

    # weightsRCP = {
    #     'S': [0.5, 0.25 ,0.25 ],
    #     'VP': [.5, .5],
    #     'RelCl': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    #     'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    #     'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    #     }

    relweight = np.array([1/(i+1) for i in range(len(relpron))])
    relweight /= np.sum(relweight)

    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    vweight = np.array([1/(i+1) for i in range(len(verbs))])
    vweight /= np.sum(vweight)

    weightsRCPZipf = {
        'S': [0.5, 0.25 ,0.25 ],
        'VP': [.5, .5],
        'RelCl': relweight,
        'N': nweight,
        'V': vweight   
        }

    # Context free grammar
    return ProbabilisticGrammar(terminals2, non_terminals2, production_rulesRCP,weightsRCPZipf)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgRCP)

def create_stimuliMD():
    number_of_verbs = 1
    number_of_nouns = 40
    number_of_adj = 1
    number_of_relpron = 1
    number_of_det = 1
    number_of_prep = 1
    number_of_monotransitive_verbs = 20
    number_of_ditransitive_verbs = 5

    #verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
    adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
    relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
    det = ['d' + str(i) for i in range(1, number_of_det+1)]
    prep = ['p' + str(i) for i in range(1, number_of_prep+1)]
    monotransitive_verbs = ['mv' + str(i) for i in range(1, number_of_monotransitive_verbs+1)]
    ditransitive_verbs = ['dv' + str(i) for i in range(1, number_of_ditransitive_verbs+1)]



    terminalsYP = flatten([monotransitive_verbs,ditransitive_verbs,nouns,adjs,relpron,det,prep])
    non_terminalsYP = ['S', 'N','NP','VP','V','rel','MV','DV','NPV','AP','PP','R','A','D','P']



    production_rulesYP = {
        'S': [['NP', 'VP']],
        'NP': [['N']],#,['D','N'],['D','AP','N'],['N','PP']],
        'VP': [['MV','NP'],['DV','NP','NP']],
        'AP': [['A'],['A','A' ] ],
        'PP': [['P','N']],
        'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
        'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
        'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
        'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
        'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
        'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)],
        'MV': [['mv' + str(i)] for i in range(1, number_of_monotransitive_verbs+1)],
        'DV': [['dv' + str(i)] for i in range(1, number_of_ditransitive_verbs+1)]
    }
    
    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    mvweight = np.array([1/(i+1) for i in range(len(monotransitive_verbs))])
    mvweight /= np.sum(mvweight)
    
    dvweight = np.array([1/(i+1) for i in range(len(ditransitive_verbs))])
    dvweight /= np.sum(mvweight)

    weightsYP = {
        'S': [1.0],
        'NP': [1.0],
        'VP': [.6,.4],
        'AP': [.75,.25 ],
        'PP': [1.0],
        'N': nweight ,#[1/number_of_nouns for i in range(1, number_of_nouns+1)],
        'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
        'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
        'D': [1/number_of_det for i in range(1, number_of_det+1)],
        'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
        'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
        'MV': mvweight,# [1/number_of_monotransitive_verbs for i in range(1, number_of_monotransitive_verbs+1)],
        'DV': dvweight,#[1/number_of_ditransitive_verbs for i in range(1, number_of_ditransitive_verbs+1)]
        }

    return ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgNVNMD)

def create_grammar_rel(
    number_of_nouns = 40,
    number_of_adj = 1,
    number_of_relpron = 3,
    number_of_det = 1,
    number_of_prep = 1,
    number_of_monotransitive_verbs = 20,
    number_of_ditransitive_verbs = 5):

    #verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
    adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
    relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
    det = ['d' + str(i) for i in range(1, number_of_det+1)]
    prep = ['p' + str(i) for i in range(1, number_of_prep+1)]
    monotransitive_verbs = ['mv' + str(i) for i in range(1, number_of_monotransitive_verbs+1)]
    ditransitive_verbs = ['dv' + str(i) for i in range(1, number_of_ditransitive_verbs+1)]



    terminalsYP = flatten([monotransitive_verbs,ditransitive_verbs,nouns,adjs,relpron,det,prep])
    non_terminalsYP = ['S', 'N','NP','VP','V','rel','MV','DV','AP','PP','A','D','P']



    production_rulesYP = {
        'S': [['NP', 'VP']],
        'NP': [['N']],#['D','N']],#['D','AP','N'],['N','PP']],
        'VP': [['MV','NP'],['MV','NP','rel','MV','NP'],['DV','NP','NP'],['DV','NP','NP','rel','MV','NP']],
        'AP': [['A'],['A','A' ] ],
        'PP': [['P','N']],
        'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
        'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
        'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
        'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
        'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)],
        'MV': [['mv' + str(i)] for i in range(1, number_of_monotransitive_verbs+1)],
        'DV': [['dv' + str(i)] for i in range(1, number_of_ditransitive_verbs+1)]
    }
    
    relweight = np.array([1/(i+1) for i in range(len(relpron))])
    relweight /= np.sum(relweight)

    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    mvweight = np.array([1/(i+1) for i in range(len(monotransitive_verbs))])
    mvweight /= np.sum(mvweight)
    
    dvweight = np.array([1/(i+1) for i in range(len(ditransitive_verbs))])
    dvweight /= np.sum(mvweight)

    weightsYP = {
        'S': [1.0],
        'NP': [1.],
        'VP': [.4,.2,.3,.1],
        'AP': [.5,.5 ],
        'PP': [1.0],
        #'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
        'N': nweight,
        'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
        'D': [1/number_of_det for i in range(1, number_of_det+1)],
        'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
        'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
        'MV': mvweight,#[1/number_of_monotransitive_verbs for i in range(1, number_of_monotransitive_verbs+1)],#
        'DV': dvweight,#[1/number_of_ditransitive_verbs for i in range(1, number_of_ditransitive_verbs+1)]#
        }

    return ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgYPredMD)


def create_stimuli_rel_test():
    number_of_verbs = 1
    number_of_nouns = 40
    number_of_adj = 1
    number_of_relpron = 1
    number_of_det = 1
    number_of_prep = 1
    number_of_monotransitive_verbs = 20
    number_of_ditransitive_verbs = 5

    verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(30, number_of_nouns+30)]
    adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
    relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
    det = ['d' + str(i) for i in range(1, number_of_det+1)]
    prep = ['p' + str(i) for i in range(1, number_of_prep+1)]
    monotransitive_verbs = ['mv' + str(i) for i in range(1, number_of_monotransitive_verbs+1)]
    ditransitive_verbs = ['dv' + str(i) for i in range(1, number_of_ditransitive_verbs+1)]



    terminalsYP = flatten([monotransitive_verbs,ditransitive_verbs,nouns,adjs,relpron,det,prep])
    non_terminalsYP = ['S', 'N','NP','VP','V','rel','MV','DV','AP','PP','A','D','P','REL']



    production_rulesYP = {
        'S': [['NP', 'VP']],
        'NP': [['N']],#['D','N']],#['D','AP','N'],['N','PP']],
        'VP': [['MV','NP'],['MV','NP','rel','VP'],['DV','NP','NP'],['DV','NP','NP','rel','VP']],
        'AP': [['A'],['A','A' ] ],
        'PP': [['P','N']],
        'N': [['n' + str(i)] for i in range(30, number_of_nouns+30)],
        'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
        'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
        'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
        'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
        'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)],
        'MV': [['mv' + str(i)] for i in range(1, number_of_monotransitive_verbs+1)],
        'DV': [['dv' + str(i)] for i in range(1, number_of_ditransitive_verbs+1)]
    }
    
    relweight = np.array([1/(i+1) for i in range(len(relpron))])
    relweight /= np.sum(relweight)

    nweight = np.array([1/(i+1) for i in range(len(nouns))])
    nweight /= np.sum(nweight)

    mvweight = np.array([1/(i+1) for i in range(len(monotransitive_verbs))])
    mvweight /= np.sum(mvweight)
    
    dvweight = np.array([1/(i+1) for i in range(len(ditransitive_verbs))])
    dvweight /= np.sum(mvweight)

    weightsYP = {
        'S': [1.0],
        'NP': [1.],
        'VP': [.4,.2,.3,.1],
        'AP': [.5,.5 ],
        'PP': [1.0],
        # 'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
        'N': nweight,
        'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
        'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
        'D': [1/number_of_det for i in range(1, number_of_det+1)],
        'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
        'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
        'MV': mvweight,#[1/number_of_monotransitive_verbs for i in range(1, number_of_monotransitive_verbs+1)],
        'DV': dvweight,#[1/number_of_ditransitive_verbs for i in range(1, number_of_ditransitive_verbs+1)]
        }

    return ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgYPredMD)


def create_stimuli_ComplNP():
    number_of_verbs = 1
    number_of_nouns = 20
    number_of_adj = 1
    number_of_relpron = 1
    number_of_det = 1
    number_of_prep = 1
    number_of_monotransitive_verbs = 5
    number_of_ditransitive_verbs = 1

    verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
    nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
    adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
    relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
    det = ['d' + str(i) for i in range(1, number_of_det+1)]
    prep = ['p' + str(i) for i in range(1, number_of_prep+1)]
    monotransitive_verbs = ['mv' + str(i) for i in range(1, number_of_monotransitive_verbs+1)]
    ditransitive_verbs = ['dv' + str(i) for i in range(1, number_of_ditransitive_verbs+1)]



    terminalsYP = flatten([monotransitive_verbs,ditransitive_verbs,nouns,adjs,relpron,det,prep])
    non_terminalsYP = ['S', 'N','NP','VP','V','rel','MV','DV','NPV','AP','PP','R','A','D','P']



    production_rulesYP = {
        'S': [['NP', 'VP']],
        'NP': [['N'],['D','N'],['D','AP','N'],['N','PP']],
        'VP': [['MV','NP'],['DV','NP','NP']],
        'AP': [['A'],['A','A' ] ],
        'PP': [['P','N']],
        'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
        'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
        'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
        'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
        'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
        'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)],
        'MV': [['mv' + str(i)] for i in range(1, number_of_monotransitive_verbs+1)],
        'DV': [['dv' + str(i)] for i in range(1, number_of_ditransitive_verbs+1)]
    }

    weightsYP = {
        'S': [1.0],
        'NP': [.75,.25/3,.25/3,.25/3],
        'VP': [.5,.5],
        'AP': [.85,.15 ],
        'PP': [1.0],
        'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
        'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
        'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
        'D': [1/number_of_det for i in range(1, number_of_det+1)],
        'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
        'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
        'MV': [1/number_of_monotransitive_verbs for i in range(1, number_of_monotransitive_verbs+1)],
        'DV': [1/number_of_ditransitive_verbs for i in range(1, number_of_ditransitive_verbs+1)]
        }

    return ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)
    #return RawInputLazy(n_sentences=n_sentences, grammar=cfgNVNMD)