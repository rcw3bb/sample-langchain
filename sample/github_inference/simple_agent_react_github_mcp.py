"""
ReAct agent module using MCP (Model Context Protocol) with GitHubModelsInferenceChatModel.

This module demonstrates the use of MCP (Model Context Protocol) for providing tools
to LangChain agents using the custom GitHubModelsInferenceChatModel, as an alternative
to the standard LangChain tool decorators.

Key differences from simple_agent_react_github_tool.py:
1. Uses MCP server (math_mcp_server.py) to provide tools via stdio transport
2. Uses MultiServerMCPClient to connect to and manage MCP servers
3. Tools run in separate process communication via MCP protocol
4. Supports distributed tool architectures and external tool servers

MCP Benefits:
- Standardized protocol for tool/context provision
- Language-agnostic tool servers
- Distributed tool architecture support
- Better isolation and security

Author: Ron Webb
Since: 1.0.0
"""

import os
import asyncio
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from .github_models_inference_chat_model import GitHubModelsInferenceChatModel


async def main() -> None:
    """
    Main function to run the MCP-based ReAct agent with GitHub Models.
    """
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    model_id = "openai/gpt-4o"

    if not github_token:
        raise ValueError("GITHUB_TOKEN is not set in the .env file.")

    print("Initializing GitHubModelsInferenceChatModel...")
    # Use the enhanced GitHubModelsInferenceChatModel with tool calling support
    llm_instance = GitHubModelsInferenceChatModel(
        api_key=github_token, model=model_id, temperature=0
    )

    # Get the absolute path to the math MCP server
    current_dir = Path(__file__).parent
    math_server_path = current_dir / "math_mcp_server.py"

    print("Initializing MCP client...")
    try:
        # Create MCP client configuration - use python directly since we're already in Poetry env
        connections: dict[str, dict[str, Any]] = {
            "math": {
                "command": "python",
                "args": [str(math_server_path.absolute())],
                "transport": "stdio",
            }
        }

        # Initialize MultiServerMCPClient with connection configuration
        # Type ignore: Library accepts dict format but type hints are more restrictive
        client = MultiServerMCPClient(connections)  # type: ignore[arg-type]

        print("Getting tools from MCP server...")
        tools = await client.get_tools()

        print(f"Found {len(tools)} tools from MCP server:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        tool_question = "Calculate 7 + 5 then multiply the result by 2."
        print(f"\nTool Question: {tool_question}")

        print(
            "Using create_react_agent with GitHubModelsInferenceChatModel and MCP tools..."
        )

        print("Creating ReAct agent...")
        agent = create_react_agent(llm_instance, tools)

        print("Starting agent execution...")
        result = await agent.ainvoke({"messages": [("user", tool_question)]})

        print("\n=== Final Answer ===")
        # Extract the final message content from the agent response
        final_message = result["messages"][-1].content
        print(f"{final_message}")

        # Check if MCP tools were actually called by looking at the log file in project root
        project_root = current_dir.parent.parent
        log_file = project_root / "mcp_tool_calls.log"
        if log_file.exists():
            print("\n=== MCP Tool Call Log ===")
            print(f"Log file location: {log_file.absolute()}")
        else:
            print("\n=== MCP Tool Call Log ===")
            print("No log file found - tools may not have been called via MCP")

    except Exception as exc:
        print(f"Error occurred: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
