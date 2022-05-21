from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction,  Configuration, \
    Cell, Fleet, Shipyard, Player, Board, Direction
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

    for shipyard in me.shipyards:
        if kore_left >= spawn_cost*max_spawn:
            action = ShipyardAction.spawn_ships(max_spawn)
            shipyard.next_action = action
        elif shipyard.ship_count > 0:
            direction = Direction.NORTH
            action = ShipyardAction.launch_fleet_with_flight_plan(2, direction.to_char())
            shipyard.next_action = action

    return me.next_actions