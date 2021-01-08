"""Test environments."""
import numpy as np
from gym.spaces import Discrete
from gym.wrappers import TimeLimit

from src.algorithms.q_learning import QLearning
from src.callbacks.stats import StatsCallback
from src.core import make_eps_greedy_policy
from src.envs.driving_agent import (
    CAR_CONDITION,
    CLOUDY,
    DRIVE_NORMAL,
    DRIVE_SLOW,
    NO_ACCIDENT,
    REWARD_NORMAL,
    REWARD_SLOW,
    ROAD_STATE,
    SUNNY,
    WEATHER,
    WET,
    DrivingAgentEnv,
    NonMarkovianDrivingAgentEnv,
)

EXPECTED_DYNAMICS = {
    0: {
        0: [(0.2, 0, 20, False), (0.6, 4, 20, False), (0.2, 9, 20, False)],
        1: [(0.2, 0, 18, False), (0.6, 4, 18, False), (0.2, 9, 18, False)],
    },
    4: {
        0: [(0.2, 0, 20, False), (0.6, 4, 20, False), (0.2, 9, 20, False)],
        1: [(0.2, 0, 18, False), (0.6, 4, 18, False), (0.2, 9, 18, False)],
    },
    9: {
        1: [(0.2, 0, 18, False), (0.6, 5, 18, False), (0.2, 9, 18, False)],
        0: [
            (0.13999999999999999, 0, 20, False),
            (0.06, 2, 0.0, True),
            (0.42, 5, 20, False),
            (0.18, 7, 0.0, True),
            (0.13999999999999999, 9, 20, False),
            (0.06, 11, 0.0, True),
        ],
    },
    5: {
        1: [(0.2, 0, 18, False), (0.6, 5, 18, False), (0.2, 9, 18, False)],
        0: [
            (0.13999999999999999, 0, 20, False),
            (0.06, 2, 0.0, True),
            (0.42, 5, 20, False),
            (0.18, 7, 0.0, True),
            (0.13999999999999999, 9, 20, False),
            (0.06, 11, 0.0, True),
        ],
    },
}


class TestDrivingAgent:
    """Test class for Driving Agent environment."""

    @classmethod
    def setup_class(cls):
        """Initialize test class."""
        cls.max_steps = 500
        cls.env = TimeLimit(DrivingAgentEnv(), max_episode_steps=cls.max_steps)

    def test_action_space(self):
        """Test action spaces."""
        assert self.env.action_space == Discrete(2)

    def test_observation_space(self):
        """Test observation spaces."""
        expected_size = len(WEATHER) * len(CAR_CONDITION) * len(ROAD_STATE)
        assert self.env.observation_space == Discrete(expected_size)

    def test_dynamics_function(self):
        """Test dynamics function."""
        assert EXPECTED_DYNAMICS == self.env.P

    def test_interaction(self):
        """
        Test consistency of interaction.

        With the random policy the environment will eventually
        get a car with an accident.
        """
        nb_steps = 0
        self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            _, reward, done, _ = self.env.step(action)
            nb_steps += 1

        # if we are here, car has got an accident
        assert reward == 0.0
        assert nb_steps != self.max_steps

    def test_interaction_optimal(self):
        """
        Test consistency of optimal interaction.

        When road is wet, drive slowly. Otherwise, drive normally.
        """
        nb_steps = 0
        state = self.env.reset()
        done = False
        while not done:
            _, _, road_state = self.env.decoder(state)
            action = DRIVE_NORMAL if road_state != WET else DRIVE_SLOW
            state, reward, done, _ = self.env.step(action)
            assert (action == DRIVE_SLOW) == (reward == REWARD_SLOW)
            assert (action == DRIVE_NORMAL) == (reward == REWARD_NORMAL)
            nb_steps += 1

        assert nb_steps == self.max_steps


def test_q_learning_learns_optimal_policy():
    """Test Q-Learning learns the optimal policy on Driving agent (Markovian)."""
    env = TimeLimit(DrivingAgentEnv(), max_episode_steps=100)
    agent = QLearning(env.observation_space, env.action_space, make_eps_greedy_policy())
    agent.train(env, nb_episodes=1000)

    stats_callback = StatsCallback()
    agent.test(env, nb_episodes=100, callbacks=[stats_callback])
    stats = stats_callback.get_stats()
    assert np.min(stats.episode_lengths) == 100


def test_q_learning_learns_suboptimal_policy_nonmarkovian():
    """Test Q-Learning learns a sub-optimal policy on non-Markovian Driving agent."""
    env = TimeLimit(NonMarkovianDrivingAgentEnv(), max_episode_steps=100)
    agent = QLearning(env.observation_space, env.action_space, make_eps_greedy_policy())
    agent.train(env, nb_episodes=1000)

    stats_callback = StatsCallback()
    agent.test(env, nb_episodes=100, callbacks=[stats_callback])
    stats = stats_callback.get_stats()

    # Q-Learning learns to don't get an accident (but suboptimally)
    assert np.min(stats.episode_lengths) == 100

    sunny_noacc = env.nm_encoder((SUNNY, NO_ACCIDENT))
    cloudy_noacc = env.nm_encoder((CLOUDY, NO_ACCIDENT))
    wet_noacc = env.nm_encoder((WET, NO_ACCIDENT))
    # best learned action when sunny: drive normal
    assert agent.choose_best_action(sunny_noacc) == DRIVE_NORMAL
    # best learned action when cloudy: drive slow (sub-optimal!)
    assert agent.choose_best_action(cloudy_noacc) == DRIVE_SLOW
    # best learned action when wet: drive slow
    assert agent.choose_best_action(wet_noacc) == DRIVE_SLOW
