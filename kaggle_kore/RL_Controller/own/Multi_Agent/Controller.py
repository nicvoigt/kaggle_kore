import numpy as np
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
# TODO Das als Paket machen, dann läuft es auch im colab und kann auf gpu
# trainiert werden.
from kaggle_kore.RL_Controller.own.helper_functions import ReplayBuffer
from kaggle_kore.RL_Controller.own import helper_functions
from kaggle_kore.RL_Controller.own.Agent.agent import Agent


class Multi_Agent_Controller:
    def __init__(self) -> None:
        self.total_train_counter = 0
        self.rl_state = dict()
        self.last_rl_state = dict()
        self.rl_names = [
            "turn",
            "kore_left",
            "kore_opp",
            "max_spawn",
            "ships_in_shipyard",
            "kore_of_fleet",
            "num_shipyards",
            "n_opp_shipyards",
            "done"]
        self.state_size = len(self.rl_names)
        self.last_rl_action = dict()
        self.agents = [Agent(self.state_size, action_size=4, lr=0.005)]
        self.obs = dict()
        self.config = dict()
        self.me = dict()
        self.num_shipyards = 1

    def get_states(self, obs, config, me, idx):
        board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = helper_functions.unbundle_stuff(
            obs, config)

        new_shipyards = num_shipyards - self.num_shipyards
        if (turn > 0) and (new_shipyards > 0) and (idx >= idx):
            for sy in range(new_shipyards):
                self.create_new_agent(idx)
        self.obs[idx] = obs
        self.config[idx] = config
        self.me[idx] = me
        self.num_shipyards = num_shipyards

        # TOD hier checken, ob es ein neues shipyard gibt. Wenn ja, dann baue
        # ein neues.

    def create_new_agent(self, idx) -> None:
        print("new base built")
        # TODO  Am besten hier später eine Kopie einfügen, die pretrained
        # ist.Oder den gleichen noch einmal reinkopieren

        # es gibt schon eine neue base, aber zuerst werden die anderen gesteuert
        # also abfragen, ob schon die neue base angesprochen wird
        next_idx = len(self.agents) - idx
        self.agents.append(Agent(self.state_size, action_size=4, lr=0.005))
        self.rl_state[next_idx] = self.rl_state[idx]

    def store_transition(self, state, action, reward, next_state, done, idx):
        self.agents[idx].store_transition(
            state, action, reward, next_state, done)

    def map_input_to_rl_state(self, obs, config, idx):
        done = 0
        board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = helper_functions.unbundle_stuff(
            obs, config)
        if (turn == 399) or (opp_shipyards == 0):
            done = 1
        ships_in_shipyard = me.shipyards[idx].ship_count
        kore_of_fleet = sum([fleet.kore for fleet in me.fleets])
        rl_state = [turn, kore_left, kore_opp, max_spawn, ships_in_shipyard,
                    kore_of_fleet, num_shipyards, opp_shipyards, done]
        return rl_state

    def make_transition(self, obs, config, idx):
        board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = helper_functions.unbundle_stuff(
            obs, config)

        reward = self.calc_reward(idx)
        if turn > 0:
            self.last_rl_state[idx] = self.rl_state[idx]
        rl_state = self.map_input_to_rl_state(obs, config, idx)

        if len(
            self.last_rl_action) != 0 and len(self.last_rl_state) == len(self.rl_state):
            self.store_transition(self.last_rl_state[idx],
                                  self.last_rl_action[idx],
                                  reward,
                                  rl_state,
                                  done=rl_state[-1],
                                  idx=idx)

        # noch schauen, dass kein zusammenhang zwischen den states besteht beim
        # resetten
        if rl_state[-1] == 1:
            pass
        self.rl_state[idx] = rl_state

        if self.total_train_counter % 1000 == 0:
            for agent in self.agents:
                agent.train()

        self.total_train_counter += 1

        return rl_state

    def choose_action(self, rl_state, obs, config, idx):

        action_raw = self.agents[idx].choose_action(state_input=rl_state)
        action_raw = np.argmax(action_raw)
        action_raw = randint(0, 3)
        action = self.map_action(action_raw, obs, config, idx)

        self.last_rl_action[idx] = action_raw
        return action

    def calc_reward(self, idx):
        # Wenn keine Aktion durchgeführt wurde, weil zb nicht genügend schiffe vorhanden waren, gibt es einen negativen reward.
        # dadurch zb eher schiffe gebaut, als eine nicht mögliche aktion durchgeführt
        # -> Welche Faktoren sind hier gleich, und welche sind anders?

        reward = 0
        if len(self.last_rl_state) == 0:        # wenn das noch nicht gefüllt wurde
            return 0

        # wenn neue schiffe gebaut wurden
        # da der rl_state nach dem calc reward auch in den last_rl state geschrieben wird,
        # ist die len bei dieser abfrage immer eins höher als wenn er schon
        # integriert wurde.
        if len(self.rl_state) > len(self.last_rl_state):
            return 0

        # TODO hier alle listen auf dictionaries umstellen
        if self.rl_state[idx][self.rl_names.index("ships_in_shipyard")] > \
                self.last_rl_state[idx][self.rl_names.index("ships_in_shipyard")]:
            reward += 100

        # wenn schiffe auf resen geschickt werden und sie kore einsammeln
        if self.rl_state[idx][self.rl_names.index("kore_of_fleet")] > \
                self.last_rl_state[idx][self.rl_names.index("kore_of_fleet")]:
            reward += 15
            # TODO: hier das gegainte implementieren

        # if the number of own shipyards increases, there will be a new agent
        new_shipyard_no = self.rl_state[idx][self.rl_names.index(
            "num_shipyards")] - self.last_rl_state[idx][self.rl_names.index("num_shipyards")]
        if new_shipyard_no > 0:
            print("new abse was build")
            reward += 500

        # reward if game ends:
        if self.rl_state[idx][self.rl_names.index("done")] == 1:
            print("Game is over.")
            if self.rl_state[idx][self.rl_names.index(
                    "n_opp_shipyards")] == 0 and self.rl_state[idx][self.rl_names.index("num_shipyards")] > 0:
                return float(reward + 100000)
            if self.rl_state[idx][self.rl_names.index(
                    "kore_left")] > self.rl_state[idx][self.rl_names.index("kore_opp")]:
                reward += 100000
                print(f"reward ist: {reward}")
            elif self.rl_state[idx][self.rl_names.index("kore_left")] < self.rl_state[self.rl_names.index("kore_opp")]:
                print("Game is lost")
                reward -= 100000

        return reward

    def map_action(self, action_raw, obs, config, idx):
        board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = helper_functions.unbundle_stuff(
            obs, config)

        # send ships to random direction
        if action_raw == 0:
            direction = Direction.random_direction()
            shipcount = me.shipyards[idx].ship_count
            if shipcount > 0:
                action = ShipyardAction.launch_fleet_with_flight_plan(
                    shipcount, direction.to_char())
            else:
                action = None
        # spawn ships
        elif action_raw == 1:
            action = ShipyardAction.spawn_ships(max_spawn)

        # create new base
        elif action_raw == 2:
            action = helper_functions.send_ships_to_create_new_base(
                shipyard=me.shipyards[idx])

        elif action_raw == 3:
            action = helper_functions.attack_opponent_base(board, idx=idx)
            # print("theoretically attacked opponent base")
            if me.shipyards[idx].ship_count >= 50:
                print("vermutlich auch angegriffen")
        return action

    def save_models(self):
        # TODO hier nochmal genau reinschauen, ist njoch nciht richtig
        # implementiert
        for agent in self.agents:
            agent.model.save(self.q_eval_model_file)
            agent.target_model.save(self.q_target_model_file)
            print('... saving models ...')
