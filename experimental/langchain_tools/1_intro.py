from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


output = multiply.invoke({"a": 2, "b": 3})
print(output)

print(multiply.name)  # multiply
print(multiply.description)  # Multiply two numbers.
print(multiply.args)
# {
# 'type': 'object',
# 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
# 'required': ['a', 'b']
# }
