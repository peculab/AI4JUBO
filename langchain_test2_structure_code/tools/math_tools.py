from langchain.tools import BaseTool, StructuredTool, tool

@tool
def multiplyTool(a: int, b:int ) -> int:
	""" Input a, b. Will return the product of a and b. """
	return a*b

@tool
def divideTool(a: int, b:int ) -> float:
	""" Input a, b. Will return a divided by b. """
	return a/b

@tool
def addTool(a: int, b:int ) -> int:
	""" Input a, b. Will return the sum of a and b. """
	return a+b

@tool
def subtractTool(a: int, b:int ) -> int:
	""" Input a, b. Will return a minus b."""
	return a-b