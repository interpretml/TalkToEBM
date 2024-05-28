"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines
"""

import inspect

from typing import Union

import t2ebm
import t2ebm.llm
from t2ebm.llm import AbstractChatModel

from t2ebm.graphs import extract_graph, graph_to_text

import t2ebm.prompts as prompts

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

###################################################################################################
# Talk to the EBM about other things than graphs.
###################################################################################################


def feature_importances_to_text(ebm):
    feature_importances = ""
    for feature_idx, feature_name in enumerate(ebm.feature_names_in_):
        feature_importances += (
            f"{feature_name}: {ebm.term_importances()[feature_idx]:.2f}\n"
        )
    return feature_importances


################################################################################################################
# Ask the LLM to perform high-level tasks directly on the EBM.
################################################################################################################


def describe_graph(
    llm: Union[AbstractChatModel, str],
    ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
    feature_index: int,
    num_sentences: int = 7,
    **kwargs,
):
    """Ask the LLM to describe a graph from an EBM, using chain-of-thought reasoning.

    This function accepts arbitrary keyword arguments that are passed to the corresponding lower-level functions.

    :param ebm:
    :param feature_index:
    :param kwargs: see llm_describe_graph
    :return: A summary of the graph in at most num_sentences sentences.
    """

    # llm setup
    llm = t2ebm.llm.setup(llm)

    # extract the graph from the EBM
    extract_kwargs = list(inspect.signature(extract_graph).parameters)
    extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    graph = extract_graph(ebm, feature_index, **extract_dict)

    # convert the graph to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    graph = graph_to_text(graph, **to_text_dict)

    # get a cot sequence of messages to describe the graph
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {k: kwargs[k] for k in dict(kwargs) if k in llm_descripe_kwargs}
    messages = prompts.describe_graph_cot(
        graph, num_sentences=num_sentences, **llm_descripe_dict
    )

    # execute the prompt
    messages = t2ebm.llm.chat_completion(llm, messages)

    # the last message contains the summary
    return messages[-1]["content"]


def describe_ebm(
    llm: Union[AbstractChatModel, str],
    ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
    num_sentences: int = 30,
    **kwargs,
):
    """Ask the LLM to describe the LLM in at most {num_sentences} sentences."""

    # llm setup
    llm = t2ebm.llm.setup(llm)

    # Note: We convert all objects to text before we prompt the LLM the first time.
    # The idea is that if there is an error processing one of the graphs, we get it before we prompt the LLM.
    feature_importances = feature_importances_to_text(ebm)

    # extract the graphs from the EBM
    extract_kwargs = list(inspect.signature(extract_graph).parameters)
    extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    graphs = []
    for feature_index in range(len(ebm.feature_names_in_)):
        graphs.append(extract_graph(ebm, feature_index, **extract_dict))

    # convert the graphs to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    graphs = [graph_to_text(graph, **to_text_dict) for graph in graphs]

    # get a cot sequence of messages to describe the graph
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {
        k: kwargs[k]
        for k in dict(kwargs)
        if k in llm_descripe_kwargs and k != "num_sentences"
    }
    messages = [
        prompts.describe_graph_cot(graph, num_sentences=7, **llm_descripe_dict)
        for graph in graphs
    ]

    # execute the prompts
    graph_descriptions = [
        t2ebm.llm.chat_completion(llm, msg)[-1]["content"] for msg in messages
    ]

    # combine the graph descriptions in a single string
    graph_descriptions = "\n\n".join(
        [
            ebm.feature_names_in_[idx] + ": " + graph_description
            for idx, graph_description in enumerate(graph_descriptions)
        ]
    )

    # print(graph_descriptions)

    # now, ask the llm to summarize the different descriptions
    llm_summarize_kwargs = list(inspect.signature(prompts.summarize_ebm).parameters)
    llm_summarize_dict = {
        k: kwargs[k] for k in dict(kwargs) if k in llm_summarize_kwargs
    }
    messages = prompts.summarize_ebm(
        feature_importances,
        graph_descriptions,
        num_sentences=num_sentences,
        **llm_summarize_dict,
    )

    # execute the prompt and return the summary
    return t2ebm.llm.chat_completion(llm, messages)[-1]["content"]
