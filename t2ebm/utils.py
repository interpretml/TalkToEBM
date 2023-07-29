"""
Misc. helper functions
"""

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import openai
import tiktoken


def num_tokens_from_string_(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


@retry(
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(10),
)
def openai_completion_query(model, messages, **kwargs):
    """Catches exceptions and retries, good for deployment / running experiments"""
    response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
    return response["choices"][0]["message"]["content"]


def openai_debug_completion_query(model, messages, **kwargs):
    """Does not catch exceptions, better for debugging"""
    response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
    return response["choices"][0]["message"]["content"]


def parse_guidance_query(query):
    """This is a utility function that parses a guidance query string into a list of messages for the openai api.

    It only works with the most simple types of queries.
    """
    messages = []
    start_tokens = ["{{#system~}}", "{{#assistant~}}", "{{#user~}}"]
    # find first occurence of any start toke in the query
    position = -1
    next_token = None
    for token in start_tokens:
        next_position = query.find(token)
        if next_position != -1 and (position == -1 or next_position < position):
            position = next_position
            next_token = token
    if next_token == start_tokens[0]:  # system
        end_pos = query.find("{{~/system}}")
        messages.append(
            {
                "role": "system",
                "content": query[position + len(start_tokens[0]) : end_pos].strip(),
            }
        )
    if next_token == start_tokens[1]:  # assistant
        end_pos = query.find("{{~/assistant}}")
        messages.append(
            {
                "role": "assistant",
                "content": query[position + len(start_tokens[1]) : end_pos].strip(),
            }
        )
    if next_token == start_tokens[2]:  # user
        end_pos = query.find("{{~/user}}")
        messages.append(
            {
                "role": "user",
                "content": query[position + len(start_tokens[2]) : end_pos].strip(),
            }
        )
    if next_token is not None and len(query[end_pos:]) > 15:
        messages.extend(parse_guidance_query(query[end_pos:]))
    return messages
