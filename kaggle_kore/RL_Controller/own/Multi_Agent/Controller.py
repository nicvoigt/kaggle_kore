import os
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
from kaggle_kore.utils.paths import result_dir


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
        self.obs = None
        self.config = None
        self.me = None
        self.num_shipyards = 1

    def check_for_new_shipyard_and_create_agent(self, num_shipyards, turn):
        new_shipyards = num_shipyards - self.num_shipyards
        if (turn > 0) and (new_shipyards > 0) and (self.num_shipyards > 0):
            for sy in range(new_shipyards):
                self.create_new_agent(self.num_shipyards + sy - 1)

        self.num_shipyards = num_shipyards

    def write_env_variable_into_class(self, obs, config):
        self.obs = obs
        self.config = config

    def check_if_game_is_over(self, num_shipyards):

        if num_shipyards == 0:
            return True
        else:
            return False

    def get_states(self, obs, config):
        board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = helper_functions.unbundle_stuff(
            obs, config)
        # if  self.check_if_game_is_over(self, num_shipyards) == True:
        #    return

        self.check_for_new_shipyard_and_create_agent(num_shipyards, turn)
        self.write_env_variable_into_class(obs, config)

    def create_new_agent(self, agent_idx) -> None:
        print("new base built")
        # TODO  Am besten hier später eine Kopie einfügen, die pretrained
        # ist.Oder den gleichen noch einmal reinkopieren

        # es gibt schon eine neue base, aber zuerst werden die anderen gesteuert
        # also abfragen, ob schon die neue base angesprochen wird

        # TODO hier ist der fehler!!!!!!!

        self.agents.append(Agent(self.state_size, action_size=4, lr=0.005))

        # tryout beim höchsten rl_state noch einen hinzufügen
        no_rl_states = max(self.rl_state.keys())
        next_agent_idx = no_rl_states + 1
        self.rl_state[next_agent_idx] = self.rl_state[agent_idx]
        self.last_rl_action[next_agent_idx] = self.last_rl_action[agent_idx]

    def store_transition(
            self,
            state,
            action,
            reward,
            next_state,
            done,
            agent_idx):
        self.agents[agent_idx].store_transition(
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

    def make_transition(self, idx):
        board, me, turn, spawn_cost, kore_left, max_spawn, kore_opp, opp_shipyards, num_shipyards = helper_functions.unbundle_stuff(
            self.obs, self.config)

        reward = self.calc_reward(idx)
        if turn > 0:
            self.last_rl_state[idx] = self.rl_state[idx]
        rl_state = self.map_input_to_rl_state(self.obs, self.config, idx)

        if len(
            self.last_rl_action) != 0 and len(
            self.last_rl_state) == len(
                self.rl_state):
            self.store_transition(self.last_rl_state[idx],
                                  self.last_rl_action[idx],
                                  reward,
                                  rl_state,
                                  done=rl_state[-1],
                                  agent_idx=idx)

        # noch schauen, dass kein zusammenhang zwischen den states besteht beim
        # resetten
        if rl_state[-1] == 1:
            pass
        self.rl_state[idx] = rl_state

        if self.total_train_counter % 100 == 0:
            self.train_agents()

        self.total_train_counter += 1

        return rl_state

    def train_agents(self):
        for agent in self.agents:
            agent.train()

    def choose_action(self, rl_state, idx):

        action_raw = self.agents[idx].choose_action(state_input=rl_state)
        action_raw = np.argmax(action_raw)
        action_raw = randint(0, 3)
        action = self.map_action(action_raw, self.obs, self.config, idx)

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

        elif action_raw == 3 and opp_shipyards != 0:
            action = helper_functions.attack_opponent_base(board, idx=idx)
            if me.shipyards[idx].ship_count >= 50:
                print("vermutlich auch angegriffen")
        else:
            # falls die action 3 nicht geht, versuche zu spaw
            action = ShipyardAction.spawn_ships(max_spawn)
        return action

    def save_models(self):
        # TODO hier nochmal genau reinschauen, ist njoch nciht richtig
        # implementiert
        for no_agent, agent in enumerate(self.agents):
            agent.model.save(
                os.path.join(
                    result_dir,
                    f"agent_no{no_agent}.h5"))
            agent.target_model.save(
                os.path.join(
                    result_dir,
                    f"agent_no{no_agent}_target.h5"))

    def save_replay_memory(self):
        for no_agent, agent in enumerate(self.agents):
            filename = os.path.join(result_dir, f"rpm_agent{no_agent}.npz")
            agent.save_replay_memory(filename)

    def save_models_and_rpm(self):
        self.save_models()
