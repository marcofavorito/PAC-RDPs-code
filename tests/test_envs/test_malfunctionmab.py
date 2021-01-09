"""Test environments."""
import numpy as np
from gym.spaces import Discrete
from gym.wrappers import TimeLimit

from src.algorithms.q_learning import QLearning
from src.callbacks.stats import StatsCallback
from src.core import make_eps_greedy_policy
from src.envs.malfunction_mab import MalfunctionMAB


class BaseTestMalfunctionMAB:
    """Base test class for MalfunctionMAB environment."""

    WINNING_PROBABILITIES = [0.8, 0.2]
    MALFUNCTIONING_ARM = 0
    K = 3
    MAX_STEPS = 100

    @classmethod
    def setup_class(cls):
        """Initialize test class."""
        env = MalfunctionMAB(cls.WINNING_PROBABILITIES, cls.K, cls.MALFUNCTIONING_ARM)
        cls.env = TimeLimit(env, max_episode_steps=cls.MAX_STEPS)

    def test_action_space(self):
        """Test action spaces."""
        assert self.env.action_space == Discrete(len(self.WINNING_PROBABILITIES))

    def test_observation_space(self):
        """Test observation spaces."""
        assert self.env.observation_space == Discrete(self.K + 1)

    def test_interaction(self):
        """
        Test consistency of interaction.

        With the random policy the environment will eventually
        get a car with an accident.
        """
        nb_steps = 0
        self.env.reset()
        done = False
        bad_action = self.MALFUNCTIONING_ARM
        while not done:
            action = self.env.action_space.sample()
            state, reward, done, _ = self.env.step(action)
            # reward can't be obtained if arm is broken
            assert not (state == (self.K + 1) and action == bad_action) or reward == 0
            nb_steps += 1

        # the environment is never done
        assert nb_steps == self.MAX_STEPS


class TestMalfunctionMAB(BaseTestMalfunctionMAB):
    """Test MalfunctionMAB."""

    WINNING_PROBABILITIES = [0.8, 0.2]
    MALFUNCTIONING_ARM = 0
    K = 3
    MAX_STEPS = 100


def test_q_learning_learns_optimal_policy():
    """Test Q-Learning learns the optimal policy on MalfunctionMAB (Markovian)."""
    max_steps = 100
    k = 3
    nb_states = k + 1
    p1, p2 = 0.7, 0.3
    env = TimeLimit(MalfunctionMAB([p1, p2], k, 0), max_episode_steps=max_steps)
    agent = QLearning(env.observation_space, env.action_space, make_eps_greedy_policy())
    agent.train(env, nb_episodes=2500)

    stats_callback = StatsCallback()
    agent.test(env, nb_episodes=500, callbacks=[stats_callback])
    stats = stats_callback.get_stats()
    expected_average = p1 * (max_steps * k / nb_states) + p2 * (max_steps / nb_states)
    actual_average = np.mean(stats.episode_rewards)
    assert np.isclose(actual_average, expected_average, rtol=0.01)
