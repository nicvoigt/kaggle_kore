from kaggle_kore.RL_Controller.own.Multi_Agent.Controller import Multi_Agent_Controller as Controller
from kaggle_kore.RL_Controller.own.Agent.agent import Agent
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Configuration, \
    Cell, Fleet, Shipyard, Player, Board, Direction
from numpy import outer
import pandas as pd
from math import floor, log
from random import randint, random
from helper_functions import unbundle_stuff


rlc = Controller()


def agent(obs, config):
    # Hier wird der state eingeladen und eine aktion zurückgegeben
    # also kann ich

    board, me, turn, spawn_cost, kore_left, max_spawn, opp_kore, opp_shipyards, num_shipyards = unbundle_stuff(
        obs, config)

    for idx, shipyard in enumerate(me.shipyards):
        # TODO beim attackieren des Gegners ist es tatsächlich relevant,
        # welches sy bzw welcher Agent gerade gesteuert wird
        rlc.get_states(obs, config, me, idx)

        # state mappen und auch transition in rl-agent speichern
        rl_state = rlc.make_transition(obs, config, idx)
        # if env is done
        # TODO get the other case when, there is only one shipyard left
        if rl_state[-1] == 1:
            break
            # env.reset(2)
        shipyard.next_action = rlc.choose_action(rl_state, obs, config, idx)

    return me.next_actions
