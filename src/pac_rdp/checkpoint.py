# -*- coding: utf-8 -*-
"""Module that includes algorithms to learn RDPs."""
from typing import cast

from pdfa_learning.pdfa.render import to_graphviz

from src.callbacks.checkpoint import Checkpoint
from src.pac_rdp.agent import PacRdpAgent


class RDPCheckpoint(Checkpoint):
    """Dump current PDFA."""

    def _run_test(self, episode, **kwargs) -> None:
        """On episode end."""
        super()._run_test(episode)
        agent = cast(PacRdpAgent, self.agent)
        ep_string = f"{episode:010d}"
        output_file = self.experiment_dir / f"pdfa-{ep_string}"
        if agent.pdfa:
            to_graphviz(
                agent.pdfa,
                char2str=lambda c: str(agent._rdp_generator.decoder(c))
                if c != -1
                else "-1",
            ).render(str(output_file))
