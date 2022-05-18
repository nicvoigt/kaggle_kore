from agent import Controller, Agent
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction,  Configuration, \
    Cell, Fleet, Shipyard, Player, Board, Direction
from numpy import outer
import pandas as pd
from math import floor, log
from random import randint, random

tc = [0, 2, 7,17,34,60,97,147,212,294]
sm = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
d = {"Turns Controlled": tc,
     "Spawn Maximum": sm}
df = pd.DataFrame(data=d)

# einfacher flightplan um die base:
simple_fp = ["NESW", "SENW", "NWSE", "SWNE"]

def calc_max_flightplan_length(num_ships):
    return floor(2 * log(num_ships)) + 1



rlc = Controller()


def unbundle_stuff(obs, config):
    board = Board(obs, config)
    me=board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    max_spawn = df[df["Turns Controlled"]<=turn]["Spawn Maximum"].values[-1]

    return board, me, turn, spawn_cost, kore_left, max_spawn


def agent(obs, config):
    # Hier wird der state eingeladen und eine aktion zurückgegeben
    # also kann ich
    
    board, me, turn, spawn_cost, kore_left, max_spawn = unbundle_stuff(obs, config)

    for idx, shipyard in enumerate(me.shipyards):
        rlc.get_states(obs, config)

        # state mappen und auch transition in rl-agent speichern
        rl_state = rlc.make_transition(obs, config)

        shipyard.next_action = rlc.choose_action(rl_state, obs, config, shipyard_idx = idx)

    if turn==150:
        rlc.agent.train()
    return me.next_actions