"""Utils for notebooks."""

import tempfile
from pathlib import Path
from IPython.core.display import display, HTML, SVG
from graphviz import Digraph

from src.pdfa import PDFA
from src.pdfa.render import to_graphviz

_default_svg_style = "display: block; margin-left: auto; margin-right: auto; width: 50%;"


def display_svgs(*filenames, style=_default_svg_style):
    svgs = [SVG(filename=f).data for f in filenames]
    joined_svgs = "".join(svgs)
    no_wrap_div = f'<div style="{style}white-space: nowrap">{joined_svgs}</div>'
    display(HTML(no_wrap_div))


def render_automaton(pdfa: PDFA, **kwargs):
    digraph = to_graphviz(pdfa, **kwargs)
    render_digraph(digraph)


def render_digraph(digraph: Digraph):
    tmp_dir = tempfile.mkdtemp()
    tmp_filepath = str(Path(tmp_dir, "output"))
    digraph.render(tmp_filepath)
    display_svgs(tmp_filepath + ".svg")
