"""Test environments."""
from gym.spaces import Discrete
from gym.wrappers import TimeLimit
from hypothesis import given
from hypothesis.strategies import floats, integers, lists

from src import NonMarkovianRotatingMAB


class BaseTestRotMAB:
    """Base test class for RotMAB environment."""

    def __init__(self, winning_probs, max_steps):
        """Initialize test class."""
        self.winning_probs = winning_probs
        self.max_steps = max_steps
        self.env = TimeLimit(
            NonMarkovianRotatingMAB(winning_probs=self.winning_probs),
            max_episode_steps=self.max_steps,
        )

    def test_action_space(self):
        """Test action spaces."""
        assert self.env.action_space == Discrete(len(self.winning_probs))

    def test_observation_space(self):
        """Test observation spaces."""
        assert self.env.observation_space == Discrete(2)

    def test_interaction(self):
        """Test interaction with Rotating MAB."""
        self.env.seed()
        state = self.env.reset()
        assert state == 0

        def assert_consistency(obs, reward):
            """Assert obs = 1 iff reward = 1."""
            positive_reward = reward > 0.0
            positive_obs = obs == 1
            assert (
                positive_reward
                and positive_obs
                or (not positive_reward and not positive_obs)
            )

        for _i in range(self.max_steps - 1):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            assert_consistency(obs, reward)
            assert not done

        # last action
        obs, reward, done, info = self.env.step(0)
        assert_consistency(obs, reward)
        assert done


@given(
    winning_probs=lists(floats(0.0, 1.0), min_size=1, max_size=200),
    max_steps=integers(min_value=1, max_value=200),
)
def test_robmabs(winning_probs, max_steps):
    """Test many instances of rotmabs."""
    test = BaseTestRotMAB(winning_probs, max_steps)

    test.test_action_space()
    test.test_observation_space()
    test.test_interaction()
