API Reference
=============

High-level API
--------------

.. automodule:: t2ebm
   :members: describe_graph, describe_ebm, feature_importances_to_text,
   :show-inheritance:

Graphs
------

.. automodule:: t2ebm.graphs
   :members: EBMGraph, extract_graph
   :show-inheritance:

Prompts
-------

.. automodule:: t2ebm.prompts
   :members: graph_system_msg, describe_graph, describe_graph_cot, summarize_ebm
   :show-inheritance:

Interace to the LLM
-------------------

.. automodule:: t2ebm.llm
   :members: AbstractChatModel, OpenAIChatModel, openai_setup, setup, chat_completion
   :show-inheritance:


