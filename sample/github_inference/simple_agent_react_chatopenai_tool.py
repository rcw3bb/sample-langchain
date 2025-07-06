"""
ReAct agent module using standard LangChain ChatOpenAI with GitHub Models endpoint.

Author: Ron Webb
Since: 1.0.0
"""

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr


@tool
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.

    Args:
        a: First number to add
        b: Second number to add

    Returns:
        The sum of a and b
    """
    print(f"Tool: Adding {a} and {b}")
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiplies two numbers and returns the result.

    Args:
        a: First number to multiply
        b: Second number to multiply

    Returns:
        The product of a and b
    """
    print(f"Tool: Multiplying {a} and {b}")
    return a * b


TOOLS = [add_numbers, multiply_numbers]


if __name__ == "__main__":
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN is not set in the .env file.")

    print("Initializing ChatOpenAI with GitHub Models endpoint...")
    # Use ChatOpenAI configured to use GitHub Models endpoint
    llm = ChatOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=SecretStr(GITHUB_TOKEN),
        model="gpt-4o",
        temperature=0,
    )

    TOOL_QUESTION = "Calculate 7 + 5 then multiply the result by 2."
    print(f"Tool Question: {TOOL_QUESTION}")

    print("Using create_react_agent with ChatOpenAI...")

    try:
        print("Creating ReAct agent...")
        agent = create_react_agent(llm, TOOLS)

        print("Starting agent execution...")
        result = agent.invoke({"messages": [("user", TOOL_QUESTION)]})

        print(f"\n=== Final Answer ===")
        # Extract the final message content from the agent response
        final_message = result["messages"][-1].content
        print(f"{final_message}")
    except Exception as exc:
        print(f"Error occurred: {exc}")
        import traceback

        traceback.print_exc()
