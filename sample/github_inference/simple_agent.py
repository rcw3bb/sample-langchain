"""
GitHubModelsInferenceLLM module with LangGraph agent implementation.

Author: Ron Webb
Since: 1.0.0
"""

import os
from typing import ClassVar, Dict, TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import Field
import requests
from langchain.llms.base import LLM
from langchain.tools import Tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


class GitHubModelsInferenceLLM(LLM):
    """
    LLM implementation for GitHub Models Inference API.
    """
    token: str = Field(..., exclude=True)
    model_id: str
    api_url: ClassVar[str] = "https://models.github.ai/inference/chat/completions"
    headers: Dict[str, str] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, __context) -> None:
        """
        Post-initialization to set headers for API requests.
        """
        object.__setattr__(
            self,
            "headers",
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the LLM.
        """
        return "github_models_inference"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        """
        Calls the GitHub Models Inference API with the given prompt.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.model_id, "messages": messages, "stream": False}
        response = requests.post(
            self.api_url, headers=self.headers, json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _agenerate(self, prompts, stop=None, run_manager=None):
        """
        Async generation is not supported.
        """
        raise NotImplementedError("Async generation not supported")


def add_numbers_tool(a: str, b: str) -> str:
    """
    Adds two numbers provided as strings and returns the result as a string.
    """
    try:
        result = float(a) + float(b)
        return str(result)
    except (ValueError, TypeError) as exc:
        return f"Error: {exc}"


ADD_TOOL = Tool(
    name="AddNumbers",
    func=lambda x: add_numbers_tool(*x.split()),
    description="Adds two numbers. Input should be two numbers separated by a space.",
)


class AgentState(TypedDict):
    """
    State for the agent, containing a list of messages.
    """
    messages: Annotated[list[BaseMessage], add_messages]


def should_continue(state: AgentState) -> str:
    """
    Determine whether to continue or end the agent execution.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        if (
            "Action: AddNumbers" in last_message.content
            and "got result:" not in last_message.content
        ):
            return "agent"
    return END


def call_model(state: AgentState, llm_instance: GitHubModelsInferenceLLM) -> Dict[str, list[BaseMessage]]:
    """
    Call the LLM to generate a response.
    """
    messages = state["messages"]
    system_message = (
        "You are a helpful assistant that can use tools to answer questions.\n\n"
        "When you need to add numbers, respond with exactly this format:\n"
        "Action: AddNumbers\nAction Input: [number1] [number2]\n\n"
        "When you have the final answer, respond normally with just the answer."
    )
    prompt_parts = [system_message]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prompt_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt = "\n".join(prompt_parts)
    response = llm_instance._call(prompt)
    if "Action: AddNumbers" in response and "Action Input:" in response:
        try:
            action_input_start = response.find("Action Input:") + len("Action Input:")
            action_input = response[action_input_start:].strip().split("\n")[0].strip()
            tool_result = add_numbers_tool(*action_input.split())
            tool_response = (
                f"I used the AddNumbers tool with input '{action_input}' and "
                f"got result: {tool_result}"
            )
            ai_message = AIMessage(content=tool_response)
        except (ValueError, TypeError) as exc:
            ai_message = AIMessage(content=f"Error using tool: {exc}")
    else:
        ai_message = AIMessage(content=response)
    return {"messages": [ai_message]}


def create_agent_graph(llm_instance: GitHubModelsInferenceLLM):
    """
    Create the LangGraph agent workflow.
    """
    def call_model_wrapper(state: AgentState):
        return call_model(state, llm_instance)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model_wrapper)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"agent": "agent", END: END}
    )
    return workflow.compile()


if __name__ == "__main__":
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    MODEL_ID = "openai/gpt-4.1"

    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN is not set in the .env file.")

    llm_instance = GitHubModelsInferenceLLM(token=GITHUB_TOKEN, model_id=MODEL_ID)
    agent_graph = create_agent_graph(llm_instance)
    QUESTION = "What is 7 plus 5? Use the AddNumbers tool."
    print(f"Question: {QUESTION}")
    try:
        initial_state = {"messages": [HumanMessage(content=QUESTION)]}
        result = agent_graph.invoke(initial_state)
        final_message = result["messages"][-1]
        print(f"Agent Answer: {final_message.content}")
    except (ValueError, TypeError, requests.RequestException) as exc:
        print(f"Error occurred: {exc}")
        print(
            "Make sure you have a valid GITHUB_TOKEN in your .env file with models:read scope."
        )
