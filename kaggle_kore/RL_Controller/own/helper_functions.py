import numpy as np
from tempfile import TemporaryFile
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


class ReplayBuffer(object):

    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def sample_by_index(self, indexes):
        assert isinstance(indexes, tuple), "Input muss tuple sein"
        states = self.state_memory[indexes[0]: indexes[1]]
        actions = self.action_memory[indexes[0]: indexes[1]]
        rewards = self.reward_memory[indexes[0]: indexes[1]]
        states_ = self.new_state_memory[indexes[0]: indexes[1]]
        terminal = self.terminal_memory[indexes[0]: indexes[1]]

        return states, actions, rewards, states_, terminal

    def save_to_local(self):
        outfile = TemporaryFile()
        np.savez(
            "testoutput.npz",
            self.state_memory,
            self.new_state_memory,
            self.action_memory,
            self.reward_memory,
            self.terminal_memory)


tc = [0, 2, 7, 17, 34, 60, 97, 147, 212, 294]
sm = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
d = {"Turns Controlled": tc,
     "Spawn Maximum": sm}
df = pd.DataFrame(data=d)

# einfacher flightplan um die base:
simple_fp = ["NESW", "SENW", "NWSE", "SWNE"]


def calc_max_flightplan_length(num_ships):
    return floor(2 * log(num_ships)) + 1


def unbundle_stuff(obs, config):
    board = Board(obs, config)
    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    max_spawn = df[df["Turns Controlled"] <= turn]["Spawn Maximum"].values[-1]
    opp = board.players[1]
    opp_kore = opp.kore
    opp_shipyards = len(opp.shipyards)
    num_shipyards = len(me.shipyards)

    return board, me, turn, spawn_cost, kore_left, max_spawn, opp_kore, opp_shipyards, num_shipyards


def send_ships_in_random_directions(no_ships_to_send):
    direction = Direction.random_direction()
    action = ShipyardAction.launch_fleet_with_flight_plan(
        no_ships_to_send, direction.to_char())
    return action


def send_ships_to_create_new_base(shipyard):
    # print("theoretisch neue base gebaut")
    dist1 = randint(5, 9)
    dir = Direction.random_direction().to_char()
    dir1s = f"{dir}{dist1}C"
    fp = dir1s
    return ShipyardAction.launch_fleet_with_flight_plan(50, fp)


def find_opponent_bases(board) -> list:
    return [board.players[1].shipyards[idx].position for idx in range(
        len(board.players[1].shipyards))]


def get_own_position(board, idx) -> list:
    return board.players[0].shipyards[idx].position


def create_flightplan_to_opponent_bases(own_pos, opp_poss):
    a = 0
    x = own_pos[0]
    y = own_pos[1]
    # first only attack the first opponent base:
    opp_poss = opp_poss[-1]
    ox = opp_poss[0]
    oy = opp_poss[1]

    # kann durch ost/west gesteuet weren
    dx = int(((x - ox)**2)**(1 / 2) - 1)
    # kann durch nord/süd gesteuert werden
    dy = int(((y - oy)**2)**(1 / 2) - 1)

    dirx = choice(["W", "E"])
    diry = choice(["N", "S"])

    # wenn die bases auf der gleichen x-koordinate sind (x-ox)==0
    if x - ox == 0:
        fp = f"{dirx}"

    elif y - oy == 0:
        fp = f"{diry}"

    else:
        fp = f"{dirx}{dx}{diry}"
    return ShipyardAction.launch_fleet_with_flight_plan(50, fp)


def attack_opponent_base(board, idx):
    # first check where the base is:
    opponent_positions = find_opponent_bases(board)
    own_position = get_own_position(board, idx)
    return create_flightplan_to_opponent_bases(
        own_pos=own_position, opp_poss=opponent_positions)
