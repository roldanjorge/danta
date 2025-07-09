"""
https://python.langchain.com/docs/how_to/tool_calling/
"""

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())


# The function name, type hints, and docstring are all part of the tool
# schema that's passed to the model. Defining good, descriptive schemas
# is an extension of prompt engineering and is an important part of
# getting models to perform well.
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


tools = [add, multiply]


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12?"

output = llm_with_tools.invoke(query)
print(output)
