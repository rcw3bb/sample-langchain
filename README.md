# Sample LangChain Project

[TOC]

A sample project demonstrating how to use LangChain for building applications with large language models (LLMs).

## Overview

This repository contains examples and sample code for working with LangChain, a framework for developing applications powered by language models. LangChain provides a standard interface for chains, integrations with other tools, and end-to-end chains for common applications.

## Features

- Basic LLM chains
- Prompt templates

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

## Hugging Face

### Basic Chain

```sh
poetry run python -m sample.hugging_face.prompt_template
```
## License

MIT

## Author

Ronaldo Webb
