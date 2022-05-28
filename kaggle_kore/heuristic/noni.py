from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Configuration, \
    Cell, Fleet, Shipyard, Player, Board, Direction
from numpy import outer
import pandas as pd
from math import floor, log
from random import randint, random
#df = pd.read_csv("./rules/spawn_rules.txt", sep="\t")
from typing import List, Type
List[Type[ShipyardAction]]

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

    return board, me, turn, spawn_cost, kore_left, max_spawn


def send_ships_in_random_directions(no_ships_to_send):
    direction = Direction.random_direction()
    action = ShipyardAction.launch_fleet_with_flight_plan(
        no_ships_to_send, direction.to_char())
    return action


def uebergeordnete_steuerung(board,
                             me,
                             turn,
                             spawn_cost,
                             kore_left,
                             max_spawn) -> List[Type[ShipyardAction]]:
    """
    Hier unter anderem eine unterteilung in Early, Middle und Late - Game einführen
    """

    game_phase = check_game_phase(turn)

    # checke game phase ->Game-phase
    # leite aktionen aus spielphase ab -> aktion
    # parameter uebergeordnete steuerung 0/1

    return [game_phase]


def check_game_phase(turn):
    if turn <= 19:
        return "early"
    elif 50 > turn > 19:
        return "middle"
    elif turn >= 50:
        return "late"


def agent(obs, config):
    board, me, turn, spawn_cost, kore_left, max_spawn = unbundle_stuff(
        obs, config)
    # leite aktionen aus spielphase ab -> aktion
    # Uebergeordnete Idee
    output_us = uebergeordnete_steuerung(
        board, me, turn, spawn_cost, kore_left, max_spawn)

    for shipyard in me.shipyards:
        """
        Um viel Kore zu minen, muss man
        1) unterschiedliche wege gehen
        2) kurze wege gehen
        3) viele basen bauen -> mehr schiffsbau möglich -> mehr schiffe -> mehr einsammeln ->geil
        """
        if output_us == ["early"]:
            possible_builds = max_spawn
            shipyard.next_action = ShipyardAction.spawn_ships(possible_builds)

        elif output_us == ["middle"]:
            if shipyard.ship_count > 0:
                shipyard.next_action = send_ships_in_random_directions(
                    no_ships_to_send=shipyard.ship_count)

    return me.next_actions
