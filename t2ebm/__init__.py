"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines
"""

__version__ = "0.0.1"

# high-level functions
from ._llm import (
    feature_importances_to_text,
    llm_describe_ebm_graph,
    llm_describe_ebm,
)
