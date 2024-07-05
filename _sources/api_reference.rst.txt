API Reference
=============

High-Level API
--------------

.. automodule:: t2ebm
   :members: describe_graph, describe_ebm, feature_importances_to_text
   :show-inheritance:

Extract graphs from EBM's and convert them to text
--------------------------------------------------

.. automodule:: t2ebm.graphs
   :members: EBMGraph, extract_graph, simplify_graph, plot_graph, graph_to_text, text_to_graph
   :show-inheritance:

Prompt templates
----------------

.. automodule:: t2ebm.prompts
   :members: graph_system_msg, describe_graph, describe_graph_cot, summarize_ebm
   :show-inheritance:

Interface to the LLM
--------------------

.. automodule:: t2ebm.llm
   :members: AbstractChatModel, OpenAIChatModel, openai_setup, chat_completion
   :show-inheritance:


