from typing import Union
from tensorflow.keras import models, layers
from random import choice
import numpy as np
from gym import Env, spaces
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import (
    Board,
    Configuration,
    Observation,
    ShipyardAction,

)


class CustomKoreEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, agent2="random") -> None:
        super().__init__()
        kore_env = make("kore_fleets", debug=True)
        self.env = kore_env.train([None, agent2])
        self.env_configuration: Configuration = kore_env.configuration
        map_size = self.env_configuration.size
        self.board: Board = None
        self.old_raw_observation = None
        self.old_observation = None
        self.ships_last_step = 0
        self.kore_last_step = 500
        self.bases_last_step = 0

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(map_size, map_size, 4), dtype=np.float64
        )
        # action_space für jeden agenten
        self.action_space = spaces.Box(
            low=0, high=1, shape=(
                2,), dtype=np.float64)

    def map_value(self, value: Union[int, float],
                  enemy: bool = False) -> float:
        MAX_NATURAL_KORE_IN_SQUARE = 500
        MAX_ASSUMED_FLEET_SIZE = 1000
        MAX_ASSUMED_KORE_IN_FLEET = 5000
        max_value = float(
            max(
                MAX_NATURAL_KORE_IN_SQUARE,
                MAX_ASSUMED_FLEET_SIZE,
                MAX_ASSUMED_KORE_IN_FLEET,
            )
        )
        val = value / max_value
        if enemy:
            return -val
        return val

    def map_reward(self, observation):
        if self.old_observation is None:
            return 0

        val_ship = 30
        val_kore = 10
        val_base = 500

        # anzahl neuer Schiffe
        r4_new_ships = len(
            self.board.current_player.fleets) - self.ships_last_step
        self.ships_last_step = len(self.board.current_player.fleets)

        # menge kore
        r4_more_kore = self.board.current_player.kore - self.kore_last_step
        self.kore_last_step = self.board.current_player.kore

        # anzahl neuer bases
        r4_more_bases = len(
            self.board.current_player.shipyard_ids) - self.bases_last_step
        self.bases_last_step = len(self.board.current_player.shipyard_ids)

        return val_ship * r4_new_ships + val_kore * \
            r4_more_kore + val_base * r4_more_bases

    def build_observation(self, raw_observation: Observation) -> np.ndarray:
        """
        Our observation space will be a matrix of size
        (map_size, map_size, 4).

        - The first layer is the kore count, which has
            its values mapped to [0, 1]

        - The second layer is the fleet size, which has
            its values mapped to [-1, 1] (negative values
            are used to represent enemy fleets)

        - The third layer represents possible places that
            the enemy fleets can be at the next turn. All
            values are either -1 (enemy can be there) or
            0 (enemy can't be there).

        - The fourth layer represents the amount of kore
            that each fleet is carrying.
        """
        # Build the Board object that will help us build the layers
        board = Board(raw_observation, self.env_configuration)

        # Building the kore layer
        kore_layer = np.array(raw_observation.kore).reshape(
            self.env_configuration.size, self.env_configuration.size
        )

        # Building the fleet layer
        fleet_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size)
        )
        # - Get fleets and shipyards on the map
        fleets = [fleet for _, fleet in board.fleets.items()]
        shipyards = [shipyard for _, shipyard in board.shipyards.items()]

        # aktuelle shipyard auf den layer bringen

        for fleet in fleets:
            # - Get the position of the fleet
            position = fleet.position
            x, y = position.x, position.y
            # - Get the size of the fleet
            size = fleet.ship_count
            # - Check if the fleet is an enemy fleet
            if fleet.player != board.current_player:
                multilpier = -1
            else:
                multilpier = 1
            # - Set the fleet layer to the size of the fleet
            fleet_layer[x, y] = multilpier * self.map_value(size)

        # - Iterate over shipyards, getting its position and size
        for shipyard in shipyards:
            # - Get the position of the shipyard
            position = shipyard.position
            x, y = position.x, position.y
            # - Get the size of the shipyard
            size = shipyard.ship_count
            # - Check if the shipyard is an enemy shipyard
            if shipyard.player == board.current_player:
                multilpier = 1
            else:
                multilpier = -1
            # - Set the fleet layer to the size of the shipyard
            fleet_layer[x, y] = multilpier * self.map_value(size)
        # Building the enemy positions layer
        enemy_positions_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size)
        )
        # - Iterate over fleets
        # - Iterate over fleets
        for fleet in fleets:
            # If fleet is ours, skip it
            if fleet.player == board.current_player:
                continue
            # - Get the position of the fleet
            position = fleet.position
            x, y = position.x, position.y
            # - Set the enemy positions layer to -1
            enemy_positions_layer[x, y] = -1

        # Building the kore layer
        kore_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size)
        )
        # - Iterate over fleets
        for fleet in fleets:
            # - Get the position of the fleet
            position = fleet.position
            x, y = position.x, position.y
            # - Get the amount of kore the fleet is carrying
            kore = fleet.kore
            # - Set the kore layer to the amount of kore
            kore_layer[x, y] = kore

        # Building our observation box
        observation = np.zeros(
            (self.env_configuration.size, self.env_configuration.size, 4)
        )
        observation[:, :, 0] = kore_layer
        observation[:, :, 1] = fleet_layer
        observation[:, :, 2] = enemy_positions_layer
        observation[:, :, 3] = kore_layer

        return observation

    def match_action(
            self,
            action_space: np.ndarray,
            idx: int) -> ShipyardAction:
        """
        This function will match the action space to a
        ShipyardAction.
        """
        # If there are no shipyards, return None
        if len(self.board.current_player.shipyards) == 0:
            return None
        action_space = np.round_(action_space[0], decimals=0).astype(int)
        # - Check if the action space is [0, 0]
        if action_space[0] == 0 and action_space[1] == 0:
            return None
        # - Check if the action space is [0, 1]
        elif action_space[0] == 0 and action_space[1] == 1:
            return ShipyardAction.spawn_ships(1)
        # - Check if the action space is [1, 0]
        elif action_space[0] == 1 and action_space[1] == 0:
            ships_in_fleet = self.board.current_player.shipyards[0].ship_count
            if ships_in_fleet == 0:
                return None
            return ShipyardAction.launch_fleet_with_flight_plan(
                self.board.current_player.shipyards[idx].ship_count,
                choice(["N", "E", "S", "W"])
            )
        # - Check if the action space is [1, 1]
        # eine base bauen
        elif action_space[0] == 1 and action_space[1] == 1:
            ships_in_fleet = int(
                self.board.current_player.shipyards[idx].ship_count)
            if ships_in_fleet > 55:
                choice(["N", "E", "S", "W"])

                return ShipyardAction.launch_fleet_with_flight_plan(
                    ships_in_fleet,
                    choice(["N5W2C", "E5N2C", "S5W2C", "W5S2C"])
                )
        else:
            raise ValueError(f"Invalid action space: {action_space}")

    def reset(self):
        """
        Resets the environment.
        """
        self.raw_observation = self.env.reset()
        obs = self.build_observation(self.raw_observation)
        return obs

    def step(self, action_space: np.ndarray):
        """
        Performs an action in the environment.
        """
        # Get the Board object and update it
        self.board = Board(self.raw_observation, self.env_configuration)
        # Sets done if no shipyards are left
        if len(self.board.current_player.shipyards) == 0:
            return np.zeros((21, 21, 4)), 0, True, {}
        # Get the action for the shipyard
        for idx, shipyard in enumerate(self.board.current_player.shipyards):

            action = self.match_action(action_space, idx)
            self.board.current_player.shipyards[idx].next_action = action
        self.raw_observation, old_reward, done, info = self.env.step(
            self.board.current_player.next_actions
        )
        observation = self.build_observation(self.raw_observation)
        reward = self.map_reward(observation)
        self.save_observations(observation)

        return observation, reward, done, info

    def save_observations(self, observation):
        self.old_observation = observation
        self.old_raw_observation = self.raw_observation


def build_env():
    return CustomKoreEnv()


def build_model():
    # TODO ggf. Model so anpassen, dass nachher auch noch ein layer mit nur (x,1)-shape eingefügt wird.
    # hier sollen die generellen statistiken rein: anzahl kore, anzahl
    # schiffe, etc
    return models.Sequential(
        [
            layers.Input(shape=(21, 21, 4)),
            layers.Conv2D(64, 8),
            layers.Activation("linear"),
            layers.Conv2D(128, 10),
            layers.Activation("linear"),
            layers.Flatten(),
            layers.Dense(32),
            layers.Activation("linear"),
            layers.Dense(2),
            layers.Activation("sigmoid")
        ]
    )


class Agent:
    def __init__(self) -> None:
        self.model = None

    def build_model(self, model_parameter):
        # TODO ggf. Model so anpassen, dass nachher auch noch ein layer mit nur (x,1)-shape eingefügt wird.
        # hier sollen die generellen statistiken rein: anzahl kore, anzahl
        # schiffe, etc
        return models.Sequential(
            [
                layers.Input(shape=(21, 21, 4)),
                layers.Conv2D(64, 8),
                layers.Activation("linear"),
                layers.Conv2D(128, 10),
                layers.Activation("linear"),
                layers.Flatten(),
                layers.Dense(32),
                layers.Activation("linear"),
                layers.Dense(2),
                layers.Activation("sigmoid")
            ]
        )
