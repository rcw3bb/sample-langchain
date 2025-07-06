"""
Math MCP Server for providing basic mathematical operations.

Author: Ron Webb
Since: 1.0.0
"""

import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

log_file = Path(__file__).parent.parent.parent / "mcp_tool_calls.log"


def log_tool_call(message: str) -> None:
    """Log tool calls to a file."""
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
            f.flush()  # Ensure immediate write to disk
    except Exception as e:
        print(f"Failed to log to file: {e}", file=sys.stderr)


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.

    Args:
        a: First number to add
        b: Second number to add

    Returns:
        The sum of a and b
    """
    message = f"MCP Tool: Adding {a} and {b}"
    log_tool_call(message)
    return a + b


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiplies two numbers and returns the result.

    Args:
        a: First number to multiply
        b: Second number to multiply

    Returns:
        The product of a and b
    """
    message = f"MCP Tool: Multiplying {a} and {b}"
    log_tool_call(message)
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
