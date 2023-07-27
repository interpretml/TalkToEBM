from interpret.glassbox._ebm._utils import convert_to_intervals

import numpy as np

from dataclasses import dataclass

import matplotlib.pyplot as plt


import typing


import scipy

from t2ebm.utils import num_tokens_from_string_


################################################################################################################
# Put individual graphs to text
# Also has a datatype for graphs and various simple operations on this datatype.
################################################################################################################


# a low-level datastructure for the graphs of explainable boosting machines
@dataclass
class EBMGraph:
    feature_name: str
    feature_type: str
    x_vals: typing.List[
        typing.Tuple[float, float]
    ]  # todo add union with categorical features
    scores: typing.List[float]
    stds: typing.List[float]


def extract_graph(
    ebm,
    feature_index,
    normalization="none",
    use_feature_bounds=True,
):
    """Extract a graph from an Explainable Boosting Machine.

    This is a low-level function. It does not return the final format in which the graph is presented to the LLM.

    The purpose of this function is to extract the graph from the interals of the EBM and return it in a format that is easy to work with.

    :param ebm:
    :param feature_index:
    :param normalization: how to normalize the graph. possible values are: 'mean', 'min', 'none'
    :param use_feature_bounds: if True, the first and last bin are min and max value of the feature stored in the EBM. If false, the first and last value are -inf and inf, respectively.
    :return: EBMGraph
    """

    # read the variables from the ebm
    feature_name = ebm.feature_names_in_[feature_index]
    feature_type = ebm.feature_types_in_[feature_index]
    scores = ebm.term_scores_[feature_index][1:-1]  # Drop missing and unknown bins
    stds = ebm.standard_deviations_[feature_index][1:-1]

    # normalize the graph
    normalization_constant = None
    if normalization == "mean":
        normalization_constant = np.mean(scores)
    elif normalization == "min":
        normalization_constant = np.min(scores)
    elif normalization == "none":
        normalization_constant = 0
    else:
        raise Exception(f"Unknown normalization {normalization}")
    scores = scores - normalization_constant

    # read the x-axis bins from the ebm
    if feature_type == "continuous":
        x_vals = convert_to_intervals(ebm.bins_[feature_index][0])
        # feature bounds apply to continuous features only
        if use_feature_bounds:
            x_vals[0] = (ebm.feature_bounds_[feature_index][0], x_vals[0][1])
            x_vals[-1] = (x_vals[-1][0], ebm.feature_bounds_[feature_index][1])
    elif feature_type == "nominal":
        x_vals = ebm.bins_[feature_index][
            0
        ]  # todo: check this transformation with Paul
        x_vals = {v - 1: k for k, v in x_vals.items()}
        x_vals = [x_vals[idx] for idx in range(len(x_vals.keys()))]
    else:
        raise Exception(
            f"Feature {feature_index} is of unknown feature_type {feature_type}."
        )
    assert len(x_vals) == len(scores), "The number of bins and scores does not match."

    return EBMGraph(feature_name, feature_type, x_vals, scores, stds)


def simplify_graph(graph: EBMGraph, min_variation_per_cent: float = 0.0):
    """Simplifies a graph. Removes redundant (flat) bins from the graph.

    With min_variation_per_cent>0 (default 0.0), the function simplifies the graph by removing bins
    that correspond to a less that min_variation_per_cent change in the score, considering the overal min/max difference of score for the feature as 100%.
    this can be useful to keep a query within the context limit. Empirically, removing changes of less than 2% simplifies graphs a lot
    in terms of the number of bins/tokens, but visually we can hardly see the difference.

    :param bins:
    :param scores:
    :return: EBMGraph. A new simplified graph.
    """
    assert graph.feature_type == "continuous", "Can only simplify continuous graphs."
    x_vals, scores, stds = graph.x_vals, graph.scores, graph.stds
    total_variation = np.max(scores) - np.min(scores)
    new_x_vals, new_scores, new_stds = [], [], []
    for idx, (b0, b1) in enumerate(x_vals):
        if idx == 0:
            new_x_vals.append((b0, b1))
            new_scores.append(scores[idx])
            new_stds.append(stds[idx])
        else:
            score_prev = new_scores[-1]
            if (
                np.abs(float(score_prev) - float(scores[idx]))
                <= total_variation * min_variation_per_cent
            ):
                # extend the previous bin to b1
                new_x_vals[-1] = (new_x_vals[-1][0], b1)
                # guarantee that the the confidence bands of the simplified graph cover the original graph as well as its confidence bands
                new_stds[-1] = max(new_stds[-1], stds[idx])
            else:
                new_x_vals.append((b0, b1))
                new_scores.append(scores[idx])
                new_stds.append(stds[idx])
    return EBMGraph(
        graph.feature_name, graph.feature_type, new_x_vals, new_scores, new_stds
    )


def plot_graph(graph: EBMGraph):
    x_vals, scores, stds = graph.x_vals, graph.scores, graph.stds
    if graph.feature_type == "continuous":
        x, y, y_lower, y_upper = [], [], [], []
        for idx, bin in enumerate(x_vals):
            if bin[0] == -np.inf or bin[1] == np.inf:
                continue
            # left part of the bin
            x.append(bin[0] + 1e-12)
            y.append(scores[idx])
            y_lower.append(scores[idx] - stds[idx])
            y_upper.append(scores[idx] + stds[idx])
            # right part of the bin
            x.append(bin[1])
            y.append(scores[idx])
            y_lower.append(scores[idx] - stds[idx])
            y_upper.append(scores[idx] + stds[idx])
        # plot
        fig = plt.figure()
        plt.plot(x, y)
        plt.fill_between(x, y_lower, y_upper, alpha=0.2)
    elif (
        graph.feature_type == "nominal"
        or graph.feature_type == "boolean"
        or graph.feature_type == "categorical"
    ):
        # plot bins for the categorical features
        fig = plt.figure()
        plt.bar(x_vals, scores, yerr=stds)
    else:
        raise Exception(f"Unknown graph feature type {graph.feature_type}.")
    plt.xlabel(graph.feature_name)
    plt.title(f"{graph.feature_name} ({graph.feature_type})")


def xy_to_json_(x_vals, y_vals):
    """convert a sequence of x_vals and y_vals to a json string"""
    # continuous features
    if isinstance(x_vals[0], tuple):
        return (
            "{"
            + ", ".join([f'"({x[0]}, {x[1]})": {y}' for x, y in zip(x_vals, y_vals)])
            + "}"
        )
    # other features
    return "{" + ", ".join([f'"{x}": {y}' for x, y in zip(x_vals, y_vals)]) + "}"


def graph_to_text(
    graph: EBMGraph,
    include_description=True,
    feature_format=None,
    x_axis_precision=None,
    y_axis_precision="auto",
    confidence_bounds=True,
    confidence_level=0.95,
    max_tokens=3000,
):
    """Convert a graph to a textual representation that can be passed to a LLM.

    This function takes care of all the different formatting issues that can arise in this process.

    The user can explicitly specify the format of the feature (continuous, cateorical, boolean), as well as the precision of the values on the x-axis and y-axis. If the user does not specify these values, the function will try to infer them from the graph.

    By default, this functions adds a short descriptive text that describes the graph to the LLM.

    This function simplifies the graph so that the textual description length is at most {max_tokens} GPT-4 tokens.
    """

    # a simple auto-detect for boolean feautres
    try:
        if (
            len(graph.x_vals) == 2
            and graph.x_vals[0].upper() == "FALSE"
            and graph.x_vals[1].upper() == "TRUE"
        ):
            feature_format = "boolean"
    except:
        pass

    # determine the feature format
    if feature_format is None:
        feature_format = graph.feature_type
    if feature_format == "nominal":
        feature_format = "categorical"

    # the description of the graph depends on the feature format
    if feature_format == "continuous":
        description_text = "This graph represents a continuous-valued feature. The keys are intervals that represent ranges where the function predicts the same value.\n\n"
    elif feature_format == "categorical":
        description_text = "This graph represents categorical feature. Each key represents a possible value that the feature can take.\n\n"
    elif feature_format == "boolean":
        description_text = "This graph represents a boolean feature. The keys are 'True' and 'False', the two possible values of the feature.\n\n"
    else:
        raise Exception(f"Unknown feature format {feature_format}")

    # simplify the graph until it fits into the max_tokens limit
    total_tokens = None
    min_variation_per_cent = 0.0
    while True:
        # simplify the graph
        simplified_graph = graph
        if feature_format == "continuous":
            simplified_graph = simplify_graph(
                graph, min_variation_per_cent=min_variation_per_cent
            )

        # confidence bounds via normal approximation
        scores = simplified_graph.scores
        if confidence_bounds:
            factor = scipy.stats.norm.interval(confidence_level, loc=0, scale=1)[1]
            lower_bounds = [
                scores[idx] - factor * simplified_graph.stds[idx]
                for idx in range(len(scores))
            ]
            upper_bounds = [
                scores[idx] + factor * simplified_graph.stds[idx]
                for idx in range(len(scores))
            ]

        # adjust the precision on the x-axis. this can be dangerous if we don't know the distribution of the data (it can impact the accuracy).
        # however, this is a necessary formatting feature because certain (old?) versions of EBMs introduce trailing digits for the xbins (e.g. age as a floating point number with 10 significant digits)
        x_vals = simplified_graph.x_vals
        if x_axis_precision is not None:
            x_vals = [
                (
                    np.round(x[0], x_axis_precision).astype(str),
                    np.round(x[1], x_axis_precision).astype(str),
                )
                for x in x_vals
            ]

        # adjust the precision on the y-axis
        if y_axis_precision == "auto":  # at least 3 significant digits
            total_variation = np.max(scores) - np.min(scores)
            y_fraction = total_variation / 100.0
            y_axis_precision = 1
            while y_fraction < 1:
                y_fraction *= 10
                y_axis_precision += 1
        scores = np.round(scores, y_axis_precision).astype(str)
        if confidence_bounds:
            lower_bounds = np.round(lower_bounds, y_axis_precision).astype(str)
            upper_bounds = np.round(upper_bounds, y_axis_precision).astype(str)

        # formatting for boolean features
        if feature_format == "boolean":
            assert (
                len(x_vals) == 2
            ), f"Requested a boolean format, but the feature has more than x-axis values: {x_vals}"
            # we assume that the left key (the lower numer, typically -1 or 0), is 'False'
            x_vals = ["False", "True"]

        # create the textual representation of the graph
        prompt = ""
        if include_description:
            prompt += description_text
        prompt += f"Feature Name: {graph.feature_name}\n"
        prompt += f"Feature Type: {feature_format}\n"
        prompt += f"Means: {xy_to_json_(x_vals, scores)}\n"
        if confidence_bounds:
            prompt += f"Lower Bounds ({confidence_level*100:.0f}%-Confidence Interval): {xy_to_json_(x_vals, lower_bounds)}\n"
            prompt += f"Upper Bounds ({confidence_level*100:.0f}%-Confidence Interval): {xy_to_json_(x_vals, upper_bounds)}\n"

        # count the number of tokens
        total_tokens = num_tokens_from_string_(prompt, "gpt-4")
        if feature_format == "continuous":
            if total_tokens > max_tokens:
                if min_variation_per_cent > 0.1:
                    raise Exception(
                        f"The graph for feature {graph.feature_name} of type {graph.feature_type} requires {total_tokens} tokens even at a simplification level of 10\%. This graph is too complex to be passed to the LLM within the loken limit of {max_tokens} tokens."
                    )
                min_variation_per_cent += 0.001
            else:
                if min_variation_per_cent > 0:
                    print(
                        f"INFO: The graph of feature {graph.feature_name} was simplified by {min_variation_per_cent * 100:.1f}%."
                    )
                return prompt
        else:
            if total_tokens > max_tokens:
                raise Exception(
                    f"The graph for feature {graph.feature_name} of type {graph.feature_type} requires {total_tokens} tokens and exceeds the token limit of {max_tokens}."
                )
            else:
                return prompt
