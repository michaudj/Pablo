# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:45:17 2025

@author: jmd01
"""

from Learner import LearnerConfig, Learner
from RawInput import RawInput, RawInputLazy
from typing import Union, Callable
import numpy as np
import matplotlib.pyplot as plt

import functools


from multiprocessing import Pool, cpu_count



class Population:
    def __init__(self, 
                 n_learners: int, 
                 config: LearnerConfig, 
                 stimuli_stream: Union[RawInput, RawInputLazy] = None,
                 stimuli_factory: Callable[[], Union[RawInput, RawInputLazy]] = None):
        self.n_learners = n_learners
        self.config = config
        self.learners = []
        self.histories = []

        # Shared input mode
        if stimuli_stream is not None:
            self.stimuli_stream = stimuli_stream
            self.shared_input = True
        # Per-learner input mode
        elif stimuli_factory is not None:
            self.stimuli_factory = stimuli_factory
            self.shared_input = False
        else:
            raise ValueError("You must provide either a stimuli_stream or a stimuli_factory.")

        self._create_learners()

    def _create_learners(self):
        for _ in range(self.n_learners):
            learner = Learner(config=self.config)
            self.learners.append(learner)

    def train_all(self):
        for learner in self.learners:
            if self.shared_input:
                learner.learn(self.stimuli_stream)
            else:
                learner.learn(self.stimuli_factory())

    def get_learning_curves(self) -> np.ndarray:
        """Returns an (n_learners x n_trials) array of success values."""
        return np.array([learner.history.success for learner in self.learners])

    def plot_average_learning_curve(self, window: int = 10,show=True):
        curves = self.get_learning_curves()
        avg_curve = np.mean(curves, axis=0)
        smoothed = np.convolve(avg_curve, np.ones(window)/window, mode='valid')
        
        if show:
            plt.figure(figsize=(10, 4))
            plt.plot(smoothed, label='Average Learning Curve')
            plt.xlabel('Trial')
            plt.ylabel('Moving Average Success')
            plt.title('Population Learning Curve')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return smoothed
    
    




def train_learner(index, config, stimuli_factory):
    learner = Learner(config=config)
    stimuli = stimuli_factory()  # Get stimuli using the factory function
    learner.learn(stimuli)
    return learner.history.success

class Population2:
    def __init__(self, 
                 n_learners: int, 
                 config: LearnerConfig, 
                 stimuli_stream: Union[RawInput, RawInputLazy] = None,
                 stimuli_factory: Callable[[], Union[RawInput, RawInputLazy]] = None):
        self.n_learners = n_learners
        self.config = config
        self.learners = []
        self.histories = []

        # Shared input mode
        if stimuli_stream is not None:
            self.stimuli_stream = stimuli_stream
            self.shared_input = True
        # Per-learner input mode
        elif stimuli_factory is not None:
            self.stimuli_factory = stimuli_factory
            self.shared_input = False
        else:
            raise ValueError("You must provide either a stimuli_stream or a stimuli_factory.")

        self._create_learners()

    def _create_learners(self):
        for _ in range(self.n_learners):
            learner = Learner(config=self.config)
            self.learners.append(learner)

    def train_all(self, use_multiprocessing: bool = False):
        if use_multiprocessing and not self.shared_input:
            with Pool(processes=min(cpu_count(), self.n_learners)) as pool:
                # Partial function to fix arguments
                train_fn = functools.partial(train_learner, config=self.config, stimuli_factory=self.stimuli_factory)
                results = pool.map(train_fn, range(self.n_learners))

            for learner, success in zip(self.learners, results):
                learner.history.success = success
        else:
            for learner in self.learners:
                stimuli = self.stimuli_stream if self.shared_input else self.stimuli_factory()
                learner.learn(stimuli)

    def get_learning_curves(self) -> np.ndarray:
        """Returns an (n_learners x n_trials) array of success values."""
        return np.array([learner.history.success for learner in self.learners])

    def plot_average_learning_curve(self, window: int = 10, show=True):
        curves = self.get_learning_curves()
        avg_curve = np.mean(curves, axis=0)
        smoothed = np.convolve(avg_curve, np.ones(window) / window, mode='valid')

        if show:
            plt.figure(figsize=(10, 4))
            plt.plot(smoothed, label='Average Learning Curve')
            plt.xlabel('Trial')
            plt.ylabel('Moving Average Success')
            plt.title('Population Learning Curve')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return smoothed


