# Sample LangChain and LangGraph Project

[TOC]

A sample project demonstrating how to use [LangChain](https://python.langchain.com) and [LangGraph](https://langchain-ai.github.io/langgraph/) for building applications with large language models (LLMs).

## Overview

This repository contains examples and sample code for working with LangChain and LangGraph, frameworks for developing applications powered by language models. LangChain provides a standard interface for chains, integrations with other tools, and end-to-end chains for common applications. LangGraph extends LangChain to enable graph-based, stateful, and multi-actor workflows for more advanced LLM applications.

## Features

- Basic LLM chains (LangChain)
- Prompt templates (LangChain)
- Graph-based workflows (LangGraph)
- Stateful and multi-actor applications (LangGraph)

## Requirements

- Python >= 3.13

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/rcw3bb/sample-langchain.git
   cd sample-langchain
   ```

2. Install the dependencies:

   > If Poetry is not yet installed, use the following command to install it:
   >
   > ```sh
   > python -m pip install poetry
   > ```

   ```sh
   poetry install
   ```

## LangChain

### Hugging Face

This section demonstrates how to use LangChain with Hugging Face models for basic LLM chains and prompt templating. It provides examples of integrating external language models and building simple chain workflows.

#### Basic Chain

```sh
poetry run python -m sample.hugging_face.prompt_template
```

## LangGraph

### GitHub Inference

This section demonstrates how to use LangGraph to build a stateful agent that interacts with GitHub's Models API. It showcases multi-actor workflows and environment variable management for secure API access.

#### Environment Variable

Ensure that a `GITHUB_TOKEN` environment variable is in your system with your GitHub Models API token:

```env
GITHUB_TOKEN=your_github_models_token_here
```

#### Basic Agent

```sh
poetry run python -m sample.github_inference.simple_agent
```

#### ReAct (Reasoning and Acting) Agents

ReAct agents combine reasoning and acting capabilities to solve problems using available tools.

##### Custom GitHub Models Implementation

This example demonstrates using a custom `GitHubModelsInferenceChatModel` class that directly integrates with GitHub's Models API. This approach provides fine-grained control over the model interaction and showcases how to build custom LangChain-compatible model implementations.

Key features:
- Custom implementation of LangChain's `BaseChatModel` interface
- Direct HTTP API calls to GitHub Models inference endpoint
- Enhanced tool calling capabilities with custom message handling
- Asynchronous execution model for better performance
- Full control over request/response formatting and error handling

```sh
poetry run python -m sample.github_inference.simple_agent_react_github_tool
```

##### ChatOpenAI Implementation

This example demonstrates using the standard LangChain `ChatOpenAI` class configured to work with GitHub Models endpoint. This approach provides compatibility with existing OpenAI-based code while leveraging GitHub's model inference service.

Key features:
- Uses `ChatOpenAI` with custom `base_url` pointing to GitHub Models endpoint
- Standard LangChain `@tool` decorators for tool definition
- Synchronous execution model
- Direct integration with existing OpenAI-compatible workflows

```sh
poetry run python -m sample.github_inference.simple_agent_react_chatopenai_tool
```

##### MCP (Model Context Protocol) Implementation

This example demonstrates using MCP (Model Context Protocol) for providing tools to ReAct agents. MCP provides a standardized protocol for tool and context provision with benefits including:

- Language-agnostic tool servers
- Distributed tool architecture support
- Better isolation and security
- Standardized protocol for tool/context provision

###### With Custom GitHub Models Implementation

```sh
poetry run python -m sample.github_inference.simple_agent_react_github_mcp
```

###### With ChatOpenAI Implementation

```sh
poetry run python -m sample.github_inference.simple_agent_react_chatopenai_mcp
```

Both implementations use:
- External MCP server (`math_mcp_server.py`) running as a separate process
- `MultiServerMCPClient` for managing MCP server connections
- Stdio transport for process communication
- Tool calls logged to `mcp_tool_calls.log` for debugging

## License

MIT

## Author

Ronaldo Webb
