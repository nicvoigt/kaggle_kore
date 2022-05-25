"""
1) Controller initialisieren
2) Environment durchlaufen lassen und regelmäßig trainieren

"""

from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction,  Configuration, Cell, Fleet, Shipyard, Player, Board
from kaggle_environments import make
import pandas as pd
from agent import Controller, unbundle_stuff
import time

start = time.time()
env = make("kore_fleets", debug=True)
config = env.configuration
rlc = Controller()
obs = env.reset(2)
obs = obs[0]["observation"]

for turn in range(399):
    board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = unbundle_stuff(obs, config)

    for idx, shipyard in enumerate(me.shipyards):
        # TODO beim attackieren des Gegners ist es tatsächlich relevant, welches sy bzw welcher Agent gerade gesteuert wird
        rlc.get_states(obs, config, me, idx)

        # state mappen und auch transition in rl-agent speichern
        rl_state = rlc.make_transition(obs, config, idx)
        # if env is done
        # TODO get the other case when, there is only one shipyard left
        if rl_state[-1] == 1:
            env.reset(2)
        shipyard.next_action = rlc.choose_action(rl_state, obs, config, idx)
    

    obs = env.step([me.next_actions, me.next_actions])
    obs = obs[0]["observation"]

end = time.time()
print(f"done. Dauer ist: {end-start}." )
