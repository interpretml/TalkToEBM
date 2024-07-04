"""
Prompts that ask the LLM to perform tasks with Graphs and EBMs.

Functions either return a string or a sequene of messages / desired responses in the OpenAI message format.
"""


def graph_system_msg(expert_description="an expert statistician and data scientist"):
    """A system message that instructs the LLM to work with the graphs of an EBM.

    Args:
        expert_description (str, optional): Description of the expert that we want the LLM to be. Defaults to "an expert statistician and data scientist".

    Returns:
        str: The system message.
    """
    return f"You are {expert_description}. You interpret global explanations produced by a Generalized Additive Model (GAM). You answer all questions to the best of your ability, relying on the graphs provided by the user, any other information you are given, and your knowledge about the real world."


def describe_graph(
    graph: str,
    graph_description="",
    dataset_description="",
    task_description="Please describe the general pattern of the graph.",
):
    """Prompt the LLM to describe a graph. This is intended to be the first prompt in a conversation about a graph.

    Args:
        graph (str): The graph to describe (in JSON format, obtained from graph_to_text).
        graph_description (str, optional): Additional description of the graph (e.g. "The y-axis of the graph depicts the probability of sucess."). Defaults to "".
        dataset_description (str, optional): Additional description of the dataset (e.g. "The dataset is a Pneumonia dataset collected by [...]"). Defaults to "".
        task_description (str, optional): A final prompt to instruct the LLM. Defaults to "Please describe the general pattern of the graph.".

    Returns:
        str: The prompt to describe the graph.
    """   
    prompt = """Below is the graph of a Generalized Additive Model (GAM). The graph is presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take.
    
The graph is provided in the following format:
    - The name of the feature depicted in the graph
    - The type of the feature (continuous, categorical, or boolean)
    - Mean values
    - Lower bounds of confidence interval (optional)
    - Upper bounds of confidence interval (optional)\n\n"""

    # the graph
    prompt += f"Here is the graph:\n\n{graph}\n\n"

    # optional graph_description
    if graph_description is not None and len(graph_description) > 0:
        prompt += f"{graph_description}\n\n"

    # optional dataset description
    if dataset_description is not None and len(dataset_description) > 0:
        prompt += f"Here is a description of the dataset that the model was trained on:\n\n{dataset_description}\n\n"

    # the task that the LLM is intended to perform
    prompt += task_description
    return prompt


def describe_graph_cot(graph, num_sentences=7, **kwargs):
    """Use chain-of-thought reasoning to elicit a description of a graph in at most {num_sentences} sentences.

    Returns:
        Messages in OpenAI format.
    """
    return [
        {"role": "system", "content": graph_system_msg()},
        {"role": "user", "content": describe_graph(graph, **kwargs)},
        {"role": "assistant", "temperature": 0.7, "max_tokens": 3000},
        {
            "role": "user",
            "content": "Great, now please study the graph carefully and highlight any regions you may find surprising or counterintuitive. You may also suggest an explanation for why this behavior is surprising, and what may have caused it.",
        },
        {"role": "assistant", "temperature": 0.7, "max_tokens": 2000},
        {
            "role": "user",
            "content": f"Thanks. Now please provide a brief, at most {num_sentences} sentence description of the graph. Be sure to include any important surprising patterns in the description. You can assume that the user knows that the graph is from a Generalized Additive Model (GAM).",
        },
        {"role": "assistant", "temperature": 0.7, "max_tokens": 2000},
    ]


def summarize_ebm(
    feature_importances: str,
    graph_descriptions: str,
    expert_description="an expert statistician and data scientist",
    dataset_description="",
    num_sentences: int = None,
):
    """Prompt the LLM to summarize a Generalized Additive Model (GAM).

    Returns:
        Messages in OpenAI format.
    """
    messages = [
        {
            "role": "system",
            "content": f"You are {expert_description}. Your task is to provide an overall summary of a Generalized Additive Model (GAM). The model consists of different graphs that contain the effect of a specific input feature. ",
        }
    ]
    user_msg = """Your task is to summarize a Generalized Additive Model (GAM). To perform this task, you will be given
    - The global feature importances of the different features in the model.
    - Summaries of the graphs for the different features in the model. There is exactly one graph for each feature in the model. """
    user_msg += f"Here are the global feature importances.\n\n{feature_importances}\n\n"
    user_msg += f"Here are the descriptions of the different graphs.\n\n{graph_descriptions}\n\n"
    if dataset_description is not None and len(dataset_description) > 0:
        user_msg += f"Here is a description of the dataset that the model was trained on.\n\n{dataset_description}\n\n"
    user_msg += """Now, please provide a summary of the model.
    
The summary should contain the most important features in the model and their effect on the outcome. Unimportant effects and features can be ignored. 
    
Pay special attention to include any surprising patterns in the summary."""
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "temperature": 0.7, "max_tokens": 3000})
    if num_sentences is not None:
        messages.append(
            {
                "role": "user",
                "content": f"Great. Now shorten the above summary to at most {num_sentences} sentences. Be sure to keep the most important information.",
            }
        )
        messages.append({"role": "assistant", "temperature": 0.7, "max_tokens": 2000})
    return messages
