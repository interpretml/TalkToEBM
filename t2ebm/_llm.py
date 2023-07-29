"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines
"""

import numpy as np


import inspect

import guidance


from t2ebm.graphs import extract_graph, graph_to_text

import t2ebm.prompts as prompts

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


def llm_describe_ebm_graph(llm, ebm, feature_index, num_sentences=7, **kwargs):
    """Ask the LLM to describe a graph from an EBM, using chain-of-thought reasoning.

    This function accepts arbitrary keyword arguments that are passed to the corresponding lower-level functions.

    :param ebm:
    :param feature_index:
    :param kwargs: see llm_describe_graph
    :return: A summary of the graph in at most num_sentences sentences.
    """

    # extract the graph from the EBM
    extract_kwargs = list(inspect.signature(extract_graph).parameters)
    extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    graph = extract_graph(ebm, feature_index, **extract_dict)

    # convert the graph to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    graph = graph_to_text(graph, **to_text_dict)

    # ask the LLM to describe the graph and execute the prompt
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {k: kwargs[k] for k in dict(kwargs) if k in llm_descripe_kwargs}
    prompt = prompts.describe_graph_cot(
        graph, num_sentences=num_sentences, **llm_descripe_dict
    )

    return guidance(prompt, llm, silent=True)()["cot_graph_description"]


def llm_describe_ebm(llm, ebm, num_sentences=30, **kwargs):
    """Ask the LLM to describe the LLM in at most {num_sentences} sentences."""

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

    # construct guidance prompts that ask the LLM to describe the graphs
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {
        k: kwargs[k]
        for k in dict(kwargs)
        if k in llm_descripe_kwargs and k != "num_sentences"
    }
    graph_prompts = [
        prompts.describe_graph_cot(graph, num_sentences=7, **llm_descripe_dict)
        for graph in graphs
    ]

    # execute the prompts
    graph_descriptions = [
        guidance(p, llm, silent=True)()["cot_graph_description"] for p in graph_prompts
    ]

    # combine the graph descriptions in a single string
    graph_descriptions = "\n\n".join(
        [
            ebm.feature_names_in_[idx] + ": " + graph_description
            for idx, graph_description in enumerate(graph_descriptions)
        ]
    )

    # now, ask the llm to summarize the different descriptions
    llm_summarize_kwargs = list(inspect.signature(prompts.summarize_ebm).parameters)
    llm_summarize_dict = {
        k: kwargs[k] for k in dict(kwargs) if k in llm_summarize_kwargs
    }
    prompt = prompts.summarize_ebm(
        feature_importances,
        graph_descriptions,
        num_sentences=num_sentences,
        **llm_summarize_dict,
    )
    return guidance(prompt, llm, silent=True)()["short_summary"]
