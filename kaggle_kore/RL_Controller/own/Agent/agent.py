import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Configuration, \
    Cell, Fleet, Shipyard, Player, Board, Direction
from numpy import outer
import pandas as pd
from math import floor, log
from random import randint, random, choice
#df = pd.read_csv("./rules/spawn_rules.txt", sep="\t")
from typing import List, Type
# TODO Das als Paket machen, dann läuft es auch im colab und kann auf gpu
# trainiert werden.
from kaggle_kore.RL_Controller.own.helper_functions import ReplayBuffer
from kaggle_kore.RL_Controller.own import helper_functions


class Agent:
    def __init__(
            self,
            state_size,
            action_size,
            lr,
            batch_size=32,
            gamma=0.9) -> None:
        self.state_size = state_size
        self.model = self.create_model2(action_size, lr)
        self.target_model = self.model
        self.memory = ReplayBuffer(10_000, [state_size])
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

    def choose_action(self, state_input):
        state_input = np.array(state_input)
        state_input = np.reshape(state_input, newshape=(1, len(state_input)))
        raw_action = self.model.predict(state_input)
        return raw_action

    def train(self):
        if self.memory.mem_cntr <= self.batch_size:
            return
        states, actions, rewards, states_, done = self.memory.sample_buffer(
            self.batch_size)
        print(rewards)

        q_eval = self.model.predict(states)

        q_target = q_eval[:]

        q_next = self.target_model.predict(states_)

        indices = np.arange(self.batch_size)
        q_target[indices, actions] = rewards + \
            self.gamma * np.max(q_next, axis=1) * (1 - done)
        self.target_model.train_on_batch(states, q_target)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def create_model2(self, action_size, lr):
        model = models.Sequential()
        model.add(layers.Dense(self.state_size))
        model.add(layers.Dense(16))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(action_size))
        model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
        return model

    def save_replay_memory(self):
        self.memory.save_to_local()
