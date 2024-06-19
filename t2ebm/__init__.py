"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines
"""

from .version import __version__

# high-level functions
from .functions import (
    feature_importances_to_text,
    describe_graph,
    describe_ebm,
)
