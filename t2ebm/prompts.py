"""
Prompts that ask the LLM to perform tasks with Graphs and EBMs.
"""


def describe_graph(
    graph: str,
    expert_description="an expert statistician and data scientist",
    y_axis_description="",
    dataset_description="",
    special_task_description="",
):
    """Prompt the LLM to describe a graph. This will often be the very first prompt in a conversation.


    :param data_desc: optional description of the data
    :param outcome_desc: description of the outcome
    :param expertise_desc: description of the desired expertise of the LLM
    :return:
    """
    # a general system prompt that instructs the LLM
    # the system prompt does not contain any specific information about the data, so it could be omitted for a model that does not support a system prompt
    system_msg = f"You are {expert_description}. You interpret global explanations produced by a generalized additive model (GAM). You answer all questions to the best of your ability, combining the data contained in the graph, any data set description you are given, and your knowledge about the real world."
    # the user message begins with an introduction to the task
    user_msg = """Below is the graph of a generalized additive model (GAM). The graph is presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take.
    
The graph is provided in the following format:
    - The name of the feature depicted in the graph
    - The type of the feature (continuous, categorical, or boolean)
    - Mean values
    - Lower bounds of confidence interval (optional)
    - Upper bounds of confidence interval (optional)\n\n"""
    # optional y axis description
    if y_axis_description is not None and len(y_axis_description) > 0:
        user_msg += f"{y_axis_description}\n\n"
    # the graph
    user_msg += f"Here is the graph:\n\n{graph}\n\n"

    # the task is to describe the graph, optionally with a special task description
    user_msg += "Please describe the general pattern of the graph."
    if special_task_description is not None and len(special_task_description) > 0:
        user_msg += f" {special_task_description}\n"

    # return in the openai message format
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def describe_graph_cot(graph, num_sentences=7, **kwargs):
    """Use chain-of-thought reasoning to elicit a description of a graph in at most {num_sentences} sentences."""
    prompt = describe_graph(graph, **kwargs)
    prompt += (
        """
{{#user~}}
Great, now please study the graph carefully and highlight any regions you may find surprising or counterintuitive. You may also suggest an explanation for why this behavior is surprising, and what may have caused it.
{{~/user}}

{{#assistant~}}
{{gen 'surprises' temperature=0.7 max_tokens=2000}}
{{~/assistant}}

{{#user~}}Thanks. Now please provide a brief, at most """
        + str(num_sentences)
        + """ sentence summary of the influence of the feature on the outcome.
{{~/user}}

{{#assistant~}}
{{gen 'cot_graph_description' temperature=0.7 max_tokens=500}}
{{~/assistant}}
"""
    )
    return prompt


def summarize_ebm(
    feature_importances: str,
    graph_descriptions: str,
    expert_description="an expert statistician and data scientist.",
    dataset_description="",
    num_sentences=None,
):
    # the system prompt
    prompt = (
        "{{#system~}}\n"
        + f"""You are {expert_description}
        
Your task is to provide an overall summary of a Generalized Additive Model (GAM). The model consists of different graphs that contain the effect of a specific input feature.

You will be given:
    - The global feature importances of the different features in the model.
    - Summaries of the graphs for the different features in the model. There is exactly one graph for each feature in the model.

    {'These inputs will be given to you by the user.' if dataset_description is None or dataset_description == '' else 'The user will first provide a general description of what the dataset is about. Then you will be given the feature importance scores and the summaries of the individual features.'} 
    
    You then provide the summary. 
    
    The summary should contain the most important features in the model and their effect on the outcome. Unimportant effects and features can be ignored. 
    
    Pay special attention to include any surprising patterns in the summary.\n
    """
        + "{{~/system}}\n"
    )

    # a user-assistant interaction where the user describes the data set
    if dataset_description is not None and len(dataset_description) > 0:
        prompt += (
            "\n{{#user~}}\n"
            + dataset_description
            + "\n{{~/user}}\n\n{{#assistant~}}\nThanks for this general description of"
            " the data set. Now please provide the global feature"
            " importances.\n{{~/assistant}}\n"
        )

    # feature importances and graph descriptions
    prompt += (
        "{{#user~}}\n Here are the global feature importaces.\n\n"
        + feature_importances
        + "\n{{~/user}}\n\n"
    )
    prompt += (
        "{{#assistant~}}\n Thanks. Now please provide the descriptions of the different"
        " graphs.\n{{~/assistant}}\n\n"
    )
    prompt += (
        "{{#user~}}\n Here are the descriptions of the different graphs.\n\n"
        + graph_descriptions
        + "\n\n Now, please provide the summary. \n{{~/user}}\n\n"
    )
    prompt += """{{#assistant~}}\n{{gen 'summary' temperature=0.7 max_tokens=2000}}\n{{~/assistant}}\n\n"""

    # if requested, summary in at most {num_sentences} sentences
    if num_sentences is not None:
        prompt += (
            "{{#user~}}\n Great. Now shorten the above summary to at most "
            + str(num_sentences)
            + " sentences. Be sure to keep the most important"
            " information.\n{{~/user}}\n\n"
        )
        prompt += """{{#assistant~}}\n{{gen 'short_summary' temperature=0.7 max_tokens=1000}}\n{{~/assistant}}\n\n"""

    return prompt
