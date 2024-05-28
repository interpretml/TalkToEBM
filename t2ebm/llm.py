"""
TalkToEBM structures conversations in a generic message format that can be executed with different LLMs.

To use a custom LLM, simply implement AbstractChatModel. 
"""

from dataclasses import dataclass
from openai import OpenAI, AzureOpenAI

import copy
import os

from typing import Union


@dataclass
class AbstractChatModel:
    def chat_completion(self, messages, temperature: float, max_tokens: int):
        """Send a query to a chat model.

        :param messages: The messages to send to the model. We use the OpenAI format.
        :param temperature: The sampling temperature.
        :param max_tokens: The maximum number of tokens to generate.

        Returns:
            str: The model response.
        """
        raise NotImplementedError


class DummyChatModel(AbstractChatModel):
    def chat_completion(self, messages, temperature: float, max_tokens: int):
        return "Hihi, this is the dummy chat model! I hope you find me useful for debugging."


class OpenAIChatModel(AbstractChatModel):
    client: OpenAI = None
    model: str = None

    def __init__(self, client, model):
        super().__init__()
        self.client = client
        self.model = model

    def chat_completion(self, messages, temperature, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=90,
        )
        # we return the completion string or "" if there is an invalid response/query
        try:
            response_content = response.choices[0].message.content
        except:
            print(f"Invalid response {response}")
            response_content = ""
        if response_content is None:
            print(f"Invalid response {response}")
            response_content = ""
        return response_content

    def __repr__(self) -> str:
        return f"{self.model}"


def openai_setup(model: str, azure: bool = False, *args, **kwargs):
    """Setup an OpenAI language model.

    :param model: The name of the model (e.g. "gpt-3.5-turbo-0613").
    :param azure: If true, use a model deployed on azure.

    This function uses the following environment variables:

    - OPENAI_API_KEY
    - OPENAI_API_ORG
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_KEY
    - AZURE_OPENAI_VERSION

    Returns:
        LLM_Interface: An LLM to work with!
    """
    if azure:  # azure deployment
        client = AzureOpenAI(
            azure_endpoint=(
                os.environ["AZURE_OPENAI_ENDPOINT"]
                if "AZURE_OPENAI_ENDPOINT" in os.environ
                else None
            ),
            api_key=(
                os.environ["AZURE_OPENAI_KEY"]
                if "AZURE_OPENAI_KEY" in os.environ
                else None
            ),
            api_version=(
                os.environ["AZURE_OPENAI_VERSION"]
                if "AZURE_OPENAI_VERSION" in os.environ
                else None
            ),
            *args,
            **kwargs,
        )
    else:  # openai api
        client = OpenAI(
            api_key=(
                os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else None
            ),
            organization=(
                os.environ["OPENAI_API_ORG"] if "OPENAI_API_ORG" in os.environ else None
            ),
            *args,
            **kwargs,
        )

    # the llm
    return OpenAIChatModel(client, model)


def setup(model: Union[AbstractChatModel, str]):
    """Setup for a chat model. If the input is a string, we assume that it is the name of an OpenAI model."""
    if isinstance(model, str):
        model = openai_setup(model)
    return model


def chat_completion(llm: Union[str, AbstractChatModel], messages):
    """Execute a sequence of user and assistant messages with an AbstractChatModel.

    Sends multiple individual messages to the AbstractChatModel.
    """
    llm = setup(llm)
    # we sequentially execute all assistant messages that do not have a content.
    messages = copy.deepcopy(messages)  # do not alter the input
    for msg_idx in range(len(messages)):
        if messages[msg_idx]["role"] == "assistant":
            if not "content" in messages[msg_idx]:
                # send message
                messages[msg_idx]["content"] = llm.chat_completion(
                    messages[:msg_idx],
                    temperature=messages[msg_idx]["temperature"],
                    max_tokens=messages[msg_idx]["max_tokens"],
                )
            # remove all keys except "role" and "content"
            keys = list(messages[msg_idx].keys())
            for k in keys:
                if not k in ["role", "content"]:
                    messages[msg_idx].pop(k)
    return messages
