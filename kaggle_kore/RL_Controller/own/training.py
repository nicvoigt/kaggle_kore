"""
1) Controller initialisieren
2) Environment durchlaufen lassen und regelmäßig trainieren

"""

from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Configuration, Cell, Fleet, Shipyard, Player, Board
from kaggle_environments import make
import pandas as pd
from helper_functions import unbundle_stuff
import time
from kaggle_kore.RL_Controller.own.Multi_Agent.Controller import Multi_Agent_Controller
#from kaggle_kore.RL_Controller.own.Single_Agent.Controller import Controller
start = time.time()
env = make("kore_fleets", debug=True)
config = env.configuration
rlc = Multi_Agent_Controller()
obs = env.reset(2)
obs = obs[0]["observation"]

for turn in range(10):
    board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = unbundle_stuff(
        obs, config)

    for idx, shipyard in enumerate(me.shipyards):
        # TODO beim attackieren des Gegners ist es tatsächlich relevant,
        # welches sy bzw welcher Agent gerade gesteuert wird
        rlc.get_states(obs, config)

        # state mappen und auch transition in rl-agent speichern
        rl_state = rlc.make_transition(idx)
        # if env is done
        # TODO get the other case when, there is only one shipyard left
        if rl_state[-1] == 1:
            rlc.train_agents()
            # HIer das Abspeichern und alles hinterlgen
            env.reset(2)
        shipyard.next_action = rlc.choose_action(rl_state, idx)

    obs = env.step([me.next_actions, me.next_actions])
    obs = obs[0]["observation"]

rlc.save_models_and_rpm()
print(rlc.agents[0].memory.sample_by_index((0, 50)))
rlc.save_models()
rlc.save_replay_memory()
