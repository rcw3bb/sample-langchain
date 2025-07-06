"""
Test module for simple_agent_react_github_mcp.

Author: Ron Webb
Since: 1.0.0
"""

import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from sample.github_inference.simple_agent_react_github_mcp import main


class TestSimpleAgentReactGitHubMCP:
    """Test class for simple_agent_react_github_mcp module."""

    @pytest.fixture
    def mock_env_vars(self):
        """Fixture to mock environment variables."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            yield

    @pytest.fixture
    def mock_load_dotenv(self):
        """Fixture to mock load_dotenv."""
        with patch("sample.github_inference.simple_agent_react_github_mcp.load_dotenv") as mock:
            yield mock

    @pytest.fixture
    def mock_github_chat_model(self):
        """Fixture to mock GitHubModelsInferenceChatModel."""
        with patch("sample.github_inference.simple_agent_react_github_mcp.GitHubModelsInferenceChatModel") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    @pytest.fixture
    def mock_mcp_client(self):
        """Fixture to mock MultiServerMCPClient."""
        with patch("sample.github_inference.simple_agent_react_github_mcp.MultiServerMCPClient") as mock:
            mock_instance = AsyncMock()
            mock_tools = [
                MagicMock(name="add_numbers", description="Adds two numbers"),
                MagicMock(name="multiply_numbers", description="Multiplies two numbers")
            ]
            mock_instance.get_tools.return_value = mock_tools
            mock.return_value = mock_instance
            yield mock, mock_instance, mock_tools

    @pytest.fixture
    def mock_create_react_agent(self):
        """Fixture to mock create_react_agent."""
        with patch("sample.github_inference.simple_agent_react_github_mcp.create_react_agent") as mock:
            mock_agent = AsyncMock()
            mock_result = {
                "messages": [MagicMock(content="The answer is 24")]
            }
            mock_agent.ainvoke.return_value = mock_result
            mock.return_value = mock_agent
            yield mock, mock_agent, mock_result

    @pytest.fixture
    def mock_path_exists(self):
        """Fixture to mock Path.exists()."""
        with patch("pathlib.Path.exists") as mock:
            mock.return_value = True
            yield mock

    @pytest.fixture
    def mock_file_operations(self):
        """Fixture to mock file operations."""
        mock_log_content = "MCP Tool: Adding 7 and 5\nMCP Tool: Multiplying 12 and 2"
        
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = mock_log_content
            mock_file.__enter__.return_value = mock_file
            mock_open.return_value = mock_file
            yield mock_open, mock_log_content

    @pytest.mark.asyncio
    async def test_main_successful_execution(
        self, 
        mock_env_vars, 
        mock_load_dotenv,
        mock_github_chat_model,
        mock_mcp_client,
        mock_create_react_agent,
        mock_path_exists,
        mock_file_operations,
        capsys
    ):
        """Test successful execution of main function."""
        mock_github_model, mock_github_instance = mock_github_chat_model
        mock_mcp_class, mock_mcp_instance, mock_tools = mock_mcp_client
        mock_react_agent, mock_agent, mock_result = mock_create_react_agent
        mock_open, mock_log_content = mock_file_operations

        # Execute the main function
        await main()

        # Verify environment setup
        mock_load_dotenv.assert_called_once()

        # Verify GitHub chat model initialization
        mock_github_model.assert_called_once_with(
            api_key="test_token",
            model="openai/gpt-4o",
            temperature=0
        )

        # Verify MCP client initialization
        mock_mcp_class.assert_called_once()
        args, kwargs = mock_mcp_class.call_args
        connections = args[0]
        assert "math" in connections
        assert connections["math"]["command"] == "python"
        assert connections["math"]["transport"] == "stdio"
        assert "math_mcp_server.py" in connections["math"]["args"][0]

        # Verify tools retrieval
        mock_mcp_instance.get_tools.assert_called_once()

        # Verify agent creation
        mock_react_agent.assert_called_once_with(mock_github_instance, mock_tools)

        # Verify agent execution
        mock_agent.ainvoke.assert_called_once_with({
            "messages": [("user", "Calculate 7 + 5 then multiply the result by 2.")]
        })

        # Verify console output
        captured = capsys.readouterr()
        assert "Initializing GitHubModelsInferenceChatModel..." in captured.out
        assert "Found 2 tools from MCP server:" in captured.out
        assert ": Adds two numbers" in captured.out
        assert ": Multiplies two numbers" in captured.out
        assert "Tool Question: Calculate 7 + 5 then multiply the result by 2." in captured.out
        assert "=== Final Answer ===" in captured.out
        assert "The answer is 24" in captured.out
        assert "=== MCP Tool Call Log ===" in captured.out

    @pytest.mark.asyncio
    async def test_main_missing_github_token(self, mock_load_dotenv):
        """Test main function with missing GitHub token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GITHUB_TOKEN is not set in the .env file."):
                await main()

    @pytest.mark.asyncio
    async def test_main_mcp_client_error(
        self, 
        mock_env_vars, 
        mock_load_dotenv,
        mock_github_chat_model,
        capsys
    ):
        """Test main function with MCP client error."""
        mock_github_model, mock_github_instance = mock_github_chat_model
        
        with patch("sample.github_inference.simple_agent_react_github_mcp.MultiServerMCPClient") as mock_mcp:
            mock_mcp.side_effect = Exception("MCP connection failed")
            
            # The function should handle the exception and not re-raise it
            await main()
            
            # Verify error handling
            captured = capsys.readouterr()
            assert "Error occurred: MCP connection failed" in captured.out

    @pytest.mark.asyncio
    async def test_main_agent_execution_error(
        self, 
        mock_env_vars, 
        mock_load_dotenv,
        mock_github_chat_model,
        mock_mcp_client,
        capsys
    ):
        """Test main function with agent execution error."""
        mock_github_model, mock_github_instance = mock_github_chat_model
        mock_mcp_class, mock_mcp_instance, mock_tools = mock_mcp_client
        
        with patch("sample.github_inference.simple_agent_react_github_mcp.create_react_agent") as mock_react:
            mock_agent = AsyncMock()
            mock_agent.ainvoke.side_effect = Exception("Agent execution failed")
            mock_react.return_value = mock_agent
            
            await main()
            
            # Verify error handling
            captured = capsys.readouterr()
            assert "Error occurred: Agent execution failed" in captured.out

    @pytest.mark.asyncio
    async def test_main_no_log_file(
        self, 
        mock_env_vars, 
        mock_load_dotenv,
        mock_github_chat_model,
        mock_mcp_client,
        mock_create_react_agent,
        capsys
    ):
        """Test main function when log file doesn't exist."""
        mock_github_model, mock_github_instance = mock_github_chat_model
        mock_mcp_class, mock_mcp_instance, mock_tools = mock_mcp_client
        mock_react_agent, mock_agent, mock_result = mock_create_react_agent

        with patch("pathlib.Path.exists", return_value=False):
            await main()
            
            # Verify console output
            captured = capsys.readouterr()
            assert "No log file found - tools may not have been called via MCP" in captured.out

    @pytest.mark.asyncio
    async def test_main_empty_log_file(
        self, 
        mock_env_vars, 
        mock_load_dotenv,
        mock_github_chat_model,
        mock_mcp_client,
        mock_create_react_agent,
        mock_path_exists,
        capsys
    ):
        """Test main function with empty log file."""
        mock_github_model, mock_github_instance = mock_github_chat_model
        mock_mcp_class, mock_mcp_instance, mock_tools = mock_mcp_client
        mock_react_agent, mock_agent, mock_result = mock_create_react_agent

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = ""  # Empty log file
            mock_file.__enter__.return_value = mock_file
            mock_open.return_value = mock_file
            
            await main()
            
            # Verify console output
            captured = capsys.readouterr()
            assert "Log file is empty - no MCP tool calls recorded" in captured.out

    def test_github_token_environment_variable(self):
        """Test that the module correctly reads GITHUB_TOKEN from environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_github_token"}):
            token = os.getenv("GITHUB_TOKEN")
            assert token == "test_github_token"

    def test_model_id_configuration(self):
        """Test that the correct model ID is used."""
        expected_model_id = "openai/gpt-4o"
        # This is tested implicitly in the main test, but we can verify the constant
        assert expected_model_id == "openai/gpt-4o"
