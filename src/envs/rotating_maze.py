"""Driving agent non-Markovian environment."""
from functools import partial
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from src.helpers.gym import DiscreteEnv, space_size

SUNNY = 0
CLOUDY = 1
RAINY = 2
WEATHER = [SUNNY, CLOUDY, RAINY]

NO_ACCIDENT = 0
ACCIDENT = 1
CAR_CONDITION = [NO_ACCIDENT, ACCIDENT]

DRY = 0
WET = 1
ROAD_STATE = [DRY, WET]

SUN_PROBABILITY = 0.2
CLOUDY_PROBABILITY = 0.6
WET_PROBABILITY = 0.2
WEATHER_PROBABILITIES = [SUN_PROBABILITY, CLOUDY_PROBABILITY, WET_PROBABILITY]

ACCIDENT_PROBABILITY = 0.3
NO_ACCIDENT_PROBABILITY = 0.7

DRIVE_NORMAL = 0
DRIVE_SLOW = 1

REWARD_NORMAL = 20
REWARD_SLOW = 18
REWARDS_BY_ACTION = [REWARD_NORMAL, REWARD_SLOW]
REWARD_ACCIDENT = 0


class DrivingAgentEnv(DiscreteEnv):
    """Driving agent environment."""

    reward_range = [REWARD_ACCIDENT, REWARD_NORMAL]

    def __init__(self):
        """Initialize the environment."""
        # (sunny, cloudy, rainy) x (noAcc, acc) x (dry, wet)
        nb_weather = len(WEATHER)
        nb_conditions = len(CAR_CONDITION)
        nb_road_status = len(ROAD_STATE)
        observation_space = MultiDiscrete((nb_weather, nb_conditions, nb_road_status))
        action_space = Discrete(2)

        self.encoder = partial(np.ravel_multi_index, dims=observation_space.nvec)
        self.decoder = partial(np.unravel_index, shape=observation_space.nvec)
        nS = space_size(observation_space)
        nA = space_size(action_space)
        P: Dict[Dict] = self._compute_dynamics()
        # initial state is: (sunny, noAcc, dry)
        ids = [1.0] + [0.0 for _ in range(1, nS)]

        super().__init__(nS, nA, P, ids)

    def _compute_dynamics(self) -> Dict:
        """Compute dynamics of the system."""
        P: Dict[Any, Dict[int, List[Tuple[float, Any, int, bool]]]] = dict()

        # from sunny/cloudy-dry, no accident.
        for current_weather in [SUNNY, CLOUDY]:
            cur_state = self.encoder((current_weather, NO_ACCIDENT, DRY))
            P[cur_state] = {}
            for a in [DRIVE_NORMAL, DRIVE_SLOW]:
                transitions = []
                for next_weather in WEATHER:
                    prob = WEATHER_PROBABILITIES[next_weather]
                    next_road_state = WET if next_weather == RAINY else DRY
                    next_state = self.encoder(
                        (next_weather, NO_ACCIDENT, next_road_state)
                    )
                    reward = REWARDS_BY_ACTION[a]
                    transition = (prob, next_state, reward, False)
                    transitions.append(transition)
                P[cur_state][a] = transitions

        # from rainy-wet
        for current_weather in [RAINY, CLOUDY]:
            for next_weather in WEATHER:
                cur_state = self.encoder((current_weather, NO_ACCIDENT, WET))
                P.setdefault(cur_state, {})
                next_weather_prob = WEATHER_PROBABILITIES[next_weather]
                # if wet, it is dry only if next weather is sunny
                is_wet = WET if next_weather != SUNNY else DRY

                # when driving slowly, it is like above
                # dynamics determined only by weather
                next_state = self.encoder((next_weather, NO_ACCIDENT, is_wet))
                P[cur_state].setdefault(DRIVE_SLOW, []).append(
                    (next_weather_prob, next_state, REWARD_SLOW, False)
                )

                # when driving normally, dynamics are determined by
                # weather probability and accident probability.
                # we add two transitions:
                # (1) one where there is no accident (probability 0.7)
                no_accident_prob = NO_ACCIDENT_PROBABILITY * next_weather_prob
                next_state = self.encoder((next_weather, NO_ACCIDENT, is_wet))
                P[cur_state].setdefault(DRIVE_NORMAL, []).append(
                    (no_accident_prob, next_state, REWARD_NORMAL, False)
                )

                # (2) one where there is accident (probability 0.3)
                accident_prob = ACCIDENT_PROBABILITY * next_weather_prob
                next_state = self.encoder((next_weather, ACCIDENT, is_wet))
                P[cur_state].setdefault(DRIVE_NORMAL, []).append(
                    (accident_prob, next_state, REWARD_ACCIDENT, True)
                )

        return P


class NonMarkovianDrivingAgentEnv(gym.Wrapper):
    """Non-Markovian version of the "Driving agent" environment."""

    def __init__(self):
        """Initialize the enviornment."""
        self._unwrapped = DrivingAgentEnv()
        super().__init__(self._unwrapped)

        # non-markovian encoder-decoders
        dims_shape = (len(WEATHER), len(CAR_CONDITION))
        self.nm_encoder = partial(np.ravel_multi_index, dims=dims_shape)
        self.nm_decoder = partial(np.unravel_index, shape=dims_shape)

        self.observation_space = Discrete(np.prod(dims_shape))

    def _process(self, observation):
        """Process observation by removing road status."""
        (weather, car_condition, road_status) = self._unwrapped.decoder(observation)
        result = self.nm_encoder((weather, car_condition))
        return result

    def reset(self, **kwargs):
        """Reset the environment."""
        state = super().reset(**kwargs)
        new_state = self._process(state)
        return new_state

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step(action)
        new_state = self._process(state)
        return new_state, reward, done, info
