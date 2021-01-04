"""This module contains a callback to collect statistics."""
import time

from src.callbacks.base import Callback
from src.helpers.stats import Stats
from src.types import AgentObservation


class StatsCallback(Callback):
    """Collect statistics about the training."""

    def __init__(
        self,
    ):
        """Initialize the history callback."""
        super().__init__()
        self._reset()
        self._reset_episode()

    def _reset(self):
        """Reset state."""
        self.episode = 0
        self.total_steps = 0
        self.episode_lengths = []
        self.episode_rewards = []
        self.timestamps = []
        self.seed = None

    def _reset_episode(self):
        """Reset state after episode."""
        self.steps = 0
        self.total_reward = 0.0

    def on_training_begin(self, **kwargs) -> None:
        """On session begin."""
        self._reset()

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end."""
        s, a, r, sp, done = agent_observation
        self.total_reward += r
        self.steps += 1
        self.total_steps += 1

    def on_episode_end(self, episode, **kwargs) -> None:
        """Handle on episode end."""
        self.save_complete()
        self._reset_episode()

    def save_complete(self):
        """Save episode statistics."""
        if self.steps != 0:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(float(self.total_reward))
            self.timestamps.append(time.time())

    def get_stats(self) -> Stats:
        """Get the statistics."""
        return Stats(
            self.episode_lengths,
            self.episode_rewards,
            self.total_steps,
            self.timestamps,
        )
