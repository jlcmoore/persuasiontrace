"""
Author: Jared Moore
Date: January, 2025

Contains scripts to run a simple PII filter over persisted messages using an LLM.
"""

import json
import logging

from openai import OpenAI

from experiment.utils import extract_chat_text, replace_json_chars

logger = logging.getLogger(__name__)


PII_PROMPT = """\
Your job is to figure out if *any* of the messages we give you \
contain personally identifiable information (PII)---\
such as a name, address, birthday, etc. \
Most messages will not contain any PII, but air on the cautious side.

Format your response as a JSON list (the same as we give you), \
simply repeating back *only* the messages which contain PII.

E.g.:
```
[
    {{'role' : '<message role>', 'content' : '<message content>'}},
]
```

Input Messages:
```
{messages}
```
"""

Messages = list[dict[str, str]]


def contains_pii(messages: Messages) -> Messages:
    """
    Queries an LLM about the given messages to see if they contain PII.

    Parameters:
    messages (str OR list[dict[str, str]]): The messages

    Returns:
    Messages: the messages in the list which contain PII, or []
    """
    if not messages:
        return []

    messages_formatted = json.dumps(messages, indent=4)
    this_prompt = PII_PROMPT.format(messages=messages_formatted)

    query_messages = [{"role": "user", "content": this_prompt}]
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=query_messages,
        temperature=0,
        max_tokens=512,
    )

    result = []
    response_text = replace_json_chars(extract_chat_text(response))
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as err:
        logger.error(f"Could not decode JSON response, {response_text}")
        logger.error(err)

    # Validation of output
    input_tuple = set(tuple(msg.items()) for msg in messages)
    validated_pii = []
    for pii_message in result:
        if tuple(pii_message.items()) in input_tuple:
            validated_pii.append(pii_message)
        else:
            logger.error(f"PII message, {pii_message}, not found in input.")
    return validated_pii


def main():
    """
    Placeholder entry point for future PII filtering over saved messages.

    Not yet implemented; kept as a stub so that the module imports cleanly
    without unused-variable warnings.
    """
    raise NotImplementedError("filter_pii main() is not yet implemented.")


if __name__ == "__main__":
    main()
