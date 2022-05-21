from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction,  Configuration, \
    Cell, Fleet, Shipyard, Player, Board, Direction
import pandas as pd
from math import floor
from random import randint, random
#df = pd.read_csv("./rules/spawn_rules.txt", sep="\t")

tc = [0, 2, 7,17,34,60,97,147,212,294]
sm = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
d = {"Turns Controlled": tc,
     "Spawn Maximum": sm}
df = pd.DataFrame(data=d)



# einfacher flightplan um die base:
simple_fp = ["NESW", "SENW", "NWSE", "SWNE"]

def agent(obs, config):
    board = Board(obs, config)
    me=board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    max_spawn = df[df["Turns Controlled"]<=turn]["Spawn Maximum"].values[-1]

    for shipyard in me.shipyards:

        # build ships
        if kore_left >= spawn_cost*max_spawn:
            action = ShipyardAction.spawn_ships(max_spawn)
            shipyard.next_action = action
        elif kore_left > spawn_cost:
            poss_builds = floor(kore_left /spawn_cost)
            action = ShipyardAction.spawn_ships(poss_builds)
            shipyard.next_action = action
            
        elif shipyard.ship_count > 0:
            
            if shipyard.ship_count > 100:
                print("neue base gebaut")
                dist1 = randint(5,9)
                dist2 = randint(0,9)

                dir1 = "S" if randint(0,1)==1 else "N"
                dir2 = "E" if randint(0,1)==1 else "W"

                dir1s = f"{dir1}{dist1}{dir2}{dist2}C"
                fp = dir1s
                action = ShipyardAction.launch_fleet_with_flight_plan(50,fp)
                shipyard.next_action = action
                return me.next_actions

            if shipyard.ship_count > 80:
                dist = 5
                fp = f"S{dist}E{dist}C"
                action = ShipyardAction.launch_fleet_with_flight_plan(50,fp)
                shipyard.next_action = action

            elif shipyard.ship_count < 5:
                direction = Direction.random_direction()
                action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count,direction.to_char())
                shipyard.next_action = action
            elif 5 < shipyard.ship_count <33:
                action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count,"NESW")
                shipyard.next_action = action
                
            elif shipyard.ship_count>33:
                qs = randint(0,5)
                fp = f"N{qs}E{qs}S{qs}W{qs}"
                action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count,fp)
                shipyard.next_action = action
            


        if turn>300:
            if shipyard.ship_count>33:
                qs = randint(0,5)
                fp = f"N{qs}E{qs}S{qs}W{qs}"
                action = ShipyardAction.launch_fleet_with_flight_plan(floor((shipyard.ship_count)/3),fp)
                shipyard.next_action = action



    return me.next_actions
