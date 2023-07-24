import guidance

################################################################################################################
# Prompts that ask the LLM to perform tasks with graphs.
# We use guidance: https://github.com/microsoft/guidance
################################################################################################################


def describe_graph(
    graph: str,
    expert_description="an expert statistician and data scientist.",
    y_axis_description="",
    special_task_description="",
    dataset_description="",
    include_assistant_response=True,
):
    """Prompt the LLM to describe a graph. This will often be the very first prompt in a conversation.


    :param data_desc: optional description of the data
    :param outcome_desc: description of the outcome
    :param expertise_desc: description of the desired expertise of the LLM
    :return:
    """

    # the system prompt
    prompt = (
        "{{#system~}}\n"
        + f"""You are {expert_description}
    
You interpret global explanations produced by a generalized additive model (GAM). GAMs produce explanations in the form of graphs that contain the effect of a specific input feature.

{'You will be given graphs from the model, and the user will ask you questions about the graphs.'  if dataset_description is None or dataset_description == '' else 'The user will first provide a general description of the dataset. Then you will be given graphs from the model, and the user will ask you questions about the graphs.'} 
    
Answer all questions to the best of your ability, combining both the data contained in the graph{', the data set description you were given, and your knowledge about the real world.' if dataset_description is not None and len(dataset_description) > 0 else ' and your knowledge about the real world.'}

Graphs will be presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take. {y_axis_description if y_axis_description is not None and len(dataset_description) > 0 else ''} 
    
The user will provide graphs in the following format:
        - The name of the feature depicted in the graph
        - The type of the feature (continuous, categorical, or boolean)
        - Mean values
        - Lower bounds of confidence interval
        - Upper bounds of confidence interval

{special_task_description}\n"""
        + "{{~/system}}\n"
    )

    # a user-assistant interaction where the user describes the data set
    if dataset_description is not None and len(dataset_description) > 0:
        prompt += (
            "\n{{#user~}}\n"
            + dataset_description
            + "\n{{~/user}}\n\n{{#assistant~}}\nThanks for this general description of the data set. Please continue and provide more information, for example about the graphs from the model.\n{{~/assistant}}\n"
        )

    # the user provides the graph and asks for a description of the patterns in the graph
    prompt += (
        "\n{{#user~}}\nConsider the following graph from the model. "
        + graph
        + "\nPlease describe the general pattern of the graph.\n{{~/user}}\n\n"
    )

    # the assistant responds
    if include_assistant_response:
        prompt += """{{#assistant~}}{{gen 'graph_description' temperature=0.7 max_tokens=2000}}{{~/assistant}}"""

    return prompt


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


def summarize_dataset_prompt(
    expert_description="an expert statistician and data scientist.",
    y_axis_description="",
    dataset_description="",
    include_assistant_response=True,
):
    # the system prompt
    prompt = f"""You are {expert_description}
        
    Your task is to provide an overall summary of the relationship between the features and the outcome in a dataset. 

    You will be given:

    - The global feature importances of the different features.
    - Summaries of the effects of the individual features.

    {'The user will first provide a general description of what the dataset is about. Then you will be given the feature importance scores and the summaries of the individual features.'  if dataset_description is None or dataset_description == '' else 'These inputs will be given to you by the user.'} 
    
    You then provide the summary. The summary should contain the most important features and their effect on the outcome. Pay special attention to surprising patterns in the data, it is important that suprirising patterns are highlighted in the summary. Unimportant effects and features can be ignored.
    """
