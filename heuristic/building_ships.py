from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction,  Configuration, \
    Cell, Fleet, Shipyard, Player, Board
import pandas as pd
df = pd.read_csv("./rules/spawn_rules.txt", sep="\t")

def agent(obs, config):
    board = Board(obs, config)
    me=board.current_player

    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    max_spawn = df[df["Turns Controlled"]<=turn]["Spawn Maximum"].values[-1]
    
    # loop through all shipyards you control
    for shipyard in me.shipyards:
        # build a ship!
        if kore_left >= spawn_cost:
            action = ShipyardAction.spawn_ships(max_spawn)
            shipyard.next_action = action

    return me.next_actions