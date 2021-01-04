"""Types for the package PDFA."""
from typing import Any, Tuple

State = Any
Action = Any
Reward = Any
Done = bool
AgentObservation = Tuple[State, Action, Reward, State, Done]
