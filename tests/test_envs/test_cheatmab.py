"""Test environments."""
import numpy as np
from gym.spaces import Discrete
from gym.wrappers import TimeLimit

from src import CheatMAB
from src.algorithms.q_learning import QLearning
from src.callbacks.stats import StatsCallback
from src.core import make_eps_greedy_policy


class BaseTestCheatMAB:
    """Base test class for MalfunctionMAB environment."""

    NB_ARMS = 2
    PATTERN = [0, 0, 1]
    MAX_STEPS = 100

    @classmethod
    def setup_class(cls):
        """Initialize test class."""
        env = CheatMAB(cls.NB_ARMS, cls.PATTERN, terminate_on_win=False)
        cls.env = TimeLimit(env, max_episode_steps=cls.MAX_STEPS)

    def test_action_space(self):
        """Test action spaces."""
        assert self.env.action_space == Discrete(self.NB_ARMS)

    def test_observation_space(self):
        """Test observation spaces."""
        assert self.env.observation_space == Discrete(len(self.PATTERN) + 1)

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
            state, reward, done, _ = self.env.step(action)
            nb_steps += 1

        # the environment is never done
        assert nb_steps == self.MAX_STEPS


class TestCheatMAB(BaseTestCheatMAB):
    """Test CheatMAB."""

    NB_ARMS = 2
    PATTERN = [0, 0, 1]
    MAX_STEPS = 100


def test_q_learning_learns_optimal_policy():
    """Test Q-Learning learns the optimal policy on MalfunctionMAB (Markovian)."""
    max_steps = 100
    nb_arms = 2
    pattern = [0, 0, 1]
    env = TimeLimit(CheatMAB(nb_arms, pattern), max_episode_steps=max_steps)
    agent = QLearning(env.observation_space, env.action_space, make_eps_greedy_policy())
    agent.train(env, nb_episodes=1000)

    stats_callback = StatsCallback()
    agent.test(env, nb_episodes=100, callbacks=[stats_callback])
    stats = stats_callback.get_stats()
    assert np.min(stats.episode_lengths) == np.max(stats.episode_lengths) == 3
    assert np.sum(stats.episode_rewards) == 1.0 * max_steps
