"""Run repeated Pablo experiments with passive diagnostics; no learning-rule changes.

Usage: import run_batch from this module, then supply make_learner and
make_stimuli callbacks (see instructions in chat). Works with Learner.py API.
"""
import csv
import json
import random
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def _family(t):
    try:
        primitives = t.get_primitives()
        if primitives and all(p == '0' for p in primitives):
            return 'zero_only'
        if '1' in primitives:
            return 'contains_one'
        return 'other'
    except Exception:
        return 'other'


def _word(chunk):
    return str(chunk)


class Observer:
    def __init__(self, learner, module, csv_file, every, seed, condition):
        self.learner, self.module = learner, module
        self.every, self.seed, self.condition = every, seed, condition
        self.file = open(csv_file, 'w', newline='', encoding='utf-8')
        self.writer = None
        self.counters = Counter()
        self.birth_one = None
        self.orig_extract = learner.wm.type_assigner.extract_good_typatory_types
        self.orig_softmax = module.softmax_choice
        self.orig_record = learner.history.record
        self.pending = {}
        self.install()

    def install(self):
        def extract(chunk, starting=False):
            candidates = self.orig_extract(chunk, starting=starting)
            side = 'left' if starting else 'right'
            self.counters[f'fallback_{side}'] += 1
            for t in candidates:
                self.counters[f'offered_{side}_{_family(t)}'] += 1
            # assign_type2 uses returned dictionary directly in softmax_choice.
            self.pending[id(candidates)] = side
            return candidates

        def softmax(weights, tau=1.0):
            result = self.orig_softmax(weights, tau=tau)
            side = self.pending.pop(id(weights), None)
            if side is not None:
                self.counters[f'chosen_{side}_{_family(result)}'] += 1
            return result

        def record(success_value, length, verbose=True):
            self.orig_record(success_value, length, verbose=False)
            n = len(self.learner.history.success)
            if n % self.every == 0:
                self.snapshot(n)

        self.learner.wm.type_assigner.extract_good_typatory_types = extract
        self.module.softmax_choice = softmax
        self.learner.history.record = record

    def snapshot(self, n):
        typatory = self.learner.ltm.typatory
        assocs = self.learner.ltm.chunk_type_associations
        seen = {t for t in typatory}
        for vals in assocs.values():
            seen.update(vals)
        if self.birth_one is None and any('1' in t.get_primitives() for t in seen):
            self.birth_one = n  # observed at snapshot resolution, not exact birth
        by_type = {family: sum(max(0., float(v)) for t, v in typatory.items()
                       if _family(t) == family)
                   for family in ('zero_only', 'contains_one', 'other')}
        strong = [(t, v) for vals in assocs.values() for t, v in vals.items() if v > self.learner.wm.type_assigner.good_type_threshold]
        strong_zero = sum(_family(t) == 'zero_only' for t, _ in strong)
        recent = self.learner.history.success[max(0, n-self.every):n]
        row = {
            'condition': self.condition, 'seed': self.seed, 'trial': n,
            'accuracy_recent': sum(recent)/len(recent) if recent else None,
            'birth_one_observed': self.birth_one or '',
            'typatory_zero_mass': by_type['zero_only'],
            'typatory_one_mass': by_type['contains_one'],
            'typatory_other_mass': by_type['other'],
            'lexical_strong_total': len(strong),
            'lexical_strong_zero_share': strong_zero/len(strong) if strong else '',
            'lexical_type_count': len(seen),
            **dict(self.counters),
        }
        if self.writer is None:
            # Fixed columns, including counters absent so far.
            columns = ['condition','seed','trial','accuracy_recent','birth_one_observed',
                       'typatory_zero_mass','typatory_one_mass','typatory_other_mass',
                       'lexical_strong_total','lexical_strong_zero_share','lexical_type_count']
            for side in ('left', 'right'):
                columns.append('fallback_'+side)
                for action in ('offered', 'chosen'):
                    for family in ('zero_only','contains_one','other'):
                        columns.append(f'{action}_{side}_{family}')
            self.writer = csv.DictWriter(self.file, fieldnames=columns)
            self.writer.writeheader()
        self.writer.writerow({k: row.get(k, 0) for k in self.writer.fieldnames})
        self.file.flush()
        self.counters.clear()

    def finish(self):
        if self.counters:
            self.snapshot(len(self.learner.history.success))
        self.learner.history.record = self.orig_record
        self.module.softmax_choice = self.orig_softmax
        self.learner.wm.type_assigner.extract_good_typatory_types = self.orig_extract
        self.file.close()


def run_batch(make_learner, make_stimuli, seeds=range(20), *, output_dir='pablo_batch', every=250, condition='typatory_on'):
    """make_learner() => fresh Learner; make_stimuli(seed) => fresh input stream.

    IMPORTANT: make_stimuli(seed) must use your EXISTING rel-only grammar.
    """
    import importlib
    import sys
    from TypeNew import Type
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        Type.prim_ID = 1  # reset the primitive generator for independent runs
        learner = make_learner()
        module = importlib.import_module(type(learner).__module__)
        if not hasattr(learner.wm, 'type_assigner'):
            raise RuntimeError('Learner must use type_on=True with a type_assigner')
        observer = Observer(learner, module, out / f'{condition}_seed_{seed:03d}.csv', every, seed, condition)
        try:
            stimuli = make_stimuli(seed)
            learner.learn(stimuli)
            n = len(learner.history.success)
            accuracy_last = (sum(learner.history.success[-1000:]) / min(1000,n)) if n else 0
            words = {}
            for chunk, associations in learner.ltm.chunk_type_associations.items():
                if associations:
                    words[str(chunk)] = {repr(t): float(v) for t, v in associations.items()}
            record = {'condition':condition,'seed':seed,'n_trials':n,'accuracy_last_1000':accuracy_last,
                      'typings':words,'typatory':{repr(t):float(v) for t,v in learner.ltm.typatory.items()}}
            (out / f'{condition}_seed_{seed:03d}_final.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            results.append({k:record[k] for k in ('condition','seed','n_trials','accuracy_last_1000')})
            with open(out/'summary.csv','w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['condition','seed','n_trials','accuracy_last_1000'])
                writer.writeheader(); writer.writerows(results)
            print(f'{condition} seed {seed}: last 1000 accuracy = {accuracy_last:.3f}',flush=True)
        finally:
            observer.finish()
    return results
