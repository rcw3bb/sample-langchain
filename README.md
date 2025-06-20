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

## License

MIT

## Author

Ronaldo Webb
