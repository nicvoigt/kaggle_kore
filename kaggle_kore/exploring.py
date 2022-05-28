from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Configuration, Cell, Fleet, Shipyard, Player, Board
from kaggle_environments import make
import pandas as pd
env = make("kore_fleets", debug=True)


env.run(["RL_Controller/own/rl_main.py", "random"])
# env.run(["heuristic/noni.py", "random"])
env.render(mode="ipython", width=1000, height=800)

