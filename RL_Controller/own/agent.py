import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction,  Configuration, \
    Cell, Fleet, Shipyard, Player, Board, Direction
from numpy import outer
import pandas as pd
from math import floor, log
from random import randint, random
#df = pd.read_csv("./rules/spawn_rules.txt", sep="\t")
from typing import List, Type
from helper_functions import ReplayBuffer

List[Type[ShipyardAction]]

tc = [0, 2, 7,17,34,60,97,147,212,294]
sm = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
d = {"Turns Controlled": tc,
     "Spawn Maximum": sm}
df = pd.DataFrame(data=d)

# einfacher flightplan um die base:
simple_fp = ["NESW", "SENW", "NWSE", "SWNE"]

def calc_max_flightplan_length(num_ships):
    return floor(2 * log(num_ships)) + 1


def unbundle_stuff(obs, config):
    board = Board(obs, config)
    me=board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    max_spawn = df[df["Turns Controlled"]<=turn]["Spawn Maximum"].values[-1]

    return board, me, turn, spawn_cost, kore_left, max_spawn



def send_ships_in_random_directions(no_ships_to_send):
    direction = Direction.random_direction()
    action = ShipyardAction.launch_fleet_with_flight_plan(no_ships_to_send,direction.to_char())
    return action

class Controller:
    def __init__(self) -> None:
        
        self.rl_state = None
        self.last_rl_state = None
        self.rl_names = ["turn", "kore_left", "max_spawn", "ships_in_shipyard", "kore_of_fleet", "done"]
        self.state_size = len(self.rl_names)
        self.last_rl_action = None
        self.agent = Agent(self.state_size, action_size=2, lr=0.005)

    def get_states(self, obs, config):
        self.obs = obs
        self.config = config

    def store_transition(self, state, action, reward, next_state, done):
        self.agent.store_transition(state, action, reward, next_state, done)
        

    def map_input_to_rl_state(self, obs, config):
        done = False
        board, me, turn, spawn_cost, kore_left, max_spawn = unbundle_stuff(obs, config)
        if turn == 399:
            done = True
        ships_in_shipyard = me.shipyards[0].ship_count
        kore_of_fleet = sum([fleet.kore for fleet in me.fleets])
        rl_state = [turn, kore_left, max_spawn, ships_in_shipyard, kore_of_fleet, done]
        return rl_state

    def make_transition(self, obs, config):
        reward = self.calc_reward()
        self.last_rl_state = self.rl_state
        rl_state = self.map_input_to_rl_state(obs, config)
        
        if self.last_rl_action != None:
            self.store_transition(self.last_rl_state, self.last_rl_action, reward, rl_state, done=rl_state[-1])
        self.rl_state = rl_state
        
        return rl_state

    def choose_action(self, rl_state, obs, config, shipyard_idx):
        action_raw = self.agent.choose_action(state_input = rl_state)
        action_raw = np.argmax(action_raw)
        action_raw = randint(0,1)
        action = self.map_action(action_raw,  obs, config, shipyard_idx)
        
        self.last_rl_action = action_raw
        return action

    def calc_reward(self):
        reward = 0
        if self.last_rl_state is None:
            return 0
        # wenn neue schiffe gebaut wurden
        if self.rl_state[self.rl_names.index("ships_in_shipyard")] > \
             self.last_rl_state[self.rl_names.index("ships_in_shipyard")]:
             reward +=100

        # wenn schiffe auf resen geschickt werden und sie kore einsammeln
        if self.rl_state[self.rl_names.index("kore_of_fleet")] > \
             self.last_rl_state[self.rl_names.index("kore_of_fleet")]:
             reward +=15
             # TODO: hier das gegainte implementieren

        return reward
        

    def map_action(self, action_raw, obs, config, shipyard_idx):
        board, me, turn, spawn_cost, kore_left, max_spawn = unbundle_stuff(obs, config)
        if action_raw == 0:
            direction = Direction.random_direction()
            test = me.shipyards[shipyard_idx].ship_count
            if test>0:
                action = ShipyardAction.launch_fleet_with_flight_plan(test ,direction.to_char())
            else: action = None
        elif action_raw == 1:
            action = ShipyardAction.spawn_ships(max_spawn)

        return action

    def save_models(self):
        self.agent.model.save(self.q_eval_model_file)
        self.agent.target_model.save(self.q_target_model_file)
        print('... saving models ...')


class Agent:
    def __init__(self, state_size, action_size, lr, batch_size=64, gamma = 0.9) -> None:
        self.state_size = state_size
        self.model = self.create_model2(action_size, lr)
        self.target_model = self.model
        self.memory = ReplayBuffer(500_000, [state_size])
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        

    def create_model(self, action_size, lr):
        inputs = layers.Input(shape=())
        layer1 = layers.Dense(4, activation="relu")(inputs)
        outputs = layers.Dense(action_size, activation="linear")(layer1)
        return keras.Model(inputs=inputs, outputs=outputs)

    def choose_action(self, state_input):
        state_input = np.array(state_input)
        state_input = np.reshape(state_input, newshape=(1, len(state_input)))
        raw_action = self.model.predict(state_input)
        return raw_action
    def train(self):
        if self.memory.mem_cntr <= self.batch_size:
            return
        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)
        print(rewards)
        
        q_eval = self.model.predict(states)

        q_target = q_eval[:]

        q_next = self.target_model.predict(states_)

        indices = np.arange(self.batch_size)
        q_target[indices, actions] = rewards + self.gamma*np.max(q_next, axis=1)*(1-done)
        self.target_model.train_on_batch(states, q_target)

        

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def create_model2(self, action_size, lr):
        model = models.Sequential()
        model.add(layers.Dense(self.state_size))
        model.add(layers.Dense(4))
        model.add(layers.Dense(2, activation='relu'))
        model.add(layers.Dense(2))
        model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
        return model
