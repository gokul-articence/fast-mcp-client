from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from typing import Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from contextlib import AsyncExitStack
import os
from dotenv import load_dotenv
import re


from abc import ABC, abstractmethod
# from anthropic import Anthropic
# from openai import OpenAI

from llms import AnthropicClient, OpenAIClient,GeminiClient

# Load environment variables
load_dotenv()

app = FastAPI()

class Query(BaseModel):
    text: str

class ServerConfig:
    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args
        self.env = env

def load_server_config_secrets(config): 
    placeholder_pattern = re.compile(r"^<(.+)>$")
    if isinstance(config, dict):
        return {k: load_server_config_secrets(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [load_server_config_secrets(item) for item in config]
    elif isinstance(config, str):
        match = placeholder_pattern.match(config)
        if match:
            env_var = match.group(1)  # Extract placeholder name
            return os.getenv(env_var, f"<{env_var}_NOT_SET>")  # Default if env variable not found
    return config

def load_server_configs(config_path: str) -> Dict[str, ServerConfig]:
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        config_data = load_server_config_secrets(config_data)

        server_configs = {}
        for server_name, server_info in config_data['mcpServers'].items():
            server_configs[server_name] = ServerConfig(
                command=server_info['command'],
                args=server_info.get('args', []),
                env=server_info.get('env')
            )
        return server_configs
    except Exception as e:
        print(f"Error loading server configs: {e}")
        raise

class MCPClientManager:
    def __init__(self, server_configs: Dict[str, ServerConfig]):
        self.server_configs = server_configs
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack: Optional[AsyncExitStack] = None
        self.tools: List[Dict] = []

    async def initialize_sessions(self):
        self.exit_stack = AsyncExitStack()
        for server_name, config in self.server_configs.items():
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env
            )
            try:
                transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = transport
                session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                self.sessions[server_name] = session
                print(f"Connected to {server_name} MCP server")
            except Exception as e:
                print(f"Failed to connect to {server_name}: {e}")

    async def load_tools(self):
        for server_name, session in self.sessions.items():
            try:
                tools_result = await session.list_tools()
                self.tools.extend([
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "server": server_name,
                        "input_schema": tool.inputSchema
                    }
                    for tool in tools_result.tools
                ])
            except Exception as e:
                print(f"Error getting tools from {server_name}: {e}")

        print(f"Connected tools: {self.tools}")

    async def shutdown(self):
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None

    async def execute_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            return f"Error executing tool: {str(e)}"

# Load server configurations from JSON file
MCP_SERVERS = load_server_configs('server_config.json')
mcp_client_manager = MCPClientManager(MCP_SERVERS)

@app.on_event("startup")
async def startup_event():
    await mcp_client_manager.initialize_sessions()
    await mcp_client_manager.load_tools()

@app.on_event("shutdown")
async def shutdown_event():
    await mcp_client_manager.shutdown()

@app.post("/query")
async def process_query(query: Query):
    try:
        # Initial message to Claude
        messages = [
            {
                "role": "user",
                "content": query.text
            }
        ]

        # Process response and handle tool calls
        responses = []

        while True:
            print(messages)
            print()

            # llm_client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"), tools=mcp_client_manager.tools)
            # llm_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), tools=mcp_client_manager.tools)
            # Convert MCP tool dicts to Gemini function declarations
            function_declarations = []
            from schema_utils import clean_openapi_schema
            for tool in mcp_client_manager.tools:
                if all(k in tool for k in ("name", "description", "input_schema")):
                    # Clean input_schema for Gemini compatibility
                    input_schema = clean_openapi_schema(tool["input_schema"])
                    function_declarations.append({
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": input_schema
                    })
            llm_client = GeminiClient(api_key=os.getenv('GEMINI_API_KEY'), function_declarations=function_declarations)

            
            # Helper to flatten any object to a string before sending to Gemini
            def flatten_to_string(obj):
                # Special case: list of dicts with 'type': 'tool_result'
                if isinstance(obj, list):
                    if all(isinstance(x, dict) and x.get('type') == 'tool_result' for x in obj):
                        return "\n".join(flatten_to_string(x.get('content', '')) for x in obj)
                    return "\n".join(flatten_to_string(x) for x in obj)
                elif isinstance(obj, dict):
                    if "text" in obj and len(obj) == 1:
                        return str(obj["text"])
                    if obj.get('type') == 'tool_result' and 'content' in obj:
                        return flatten_to_string(obj['content'])
                    return "\n".join(flatten_to_string(v) for v in obj.values())
                else:
                    return str(obj)

            # Gemini expects messages as a list of strings or list of {'text': ...}
            gemini_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    gemini_messages.append(flatten_to_string(msg["content"]))
                else:
                    gemini_messages.append(flatten_to_string(msg))
            response = await llm_client.create_message(gemini_messages)
            print(response)
            print()

            parsed_response = llm_client.parse_response(response)

            print(parsed_response)
            print()

            # Append Gemini-compatible tool result message using helper defined above
            if parsed_response.content and isinstance(parsed_response.content, dict) and parsed_response.content.get("content"):
                content = parsed_response.content["content"]
                messages.append(flatten_to_string(content))
            else:
                messages.append(flatten_to_string(parsed_response.content))
            if parsed_response.text_content:
                responses.extend(parsed_response.text_content)

            if not parsed_response.tool_calls: 
                break
            
            for tool_call in parsed_response.tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.input

                # Find the correct server name for the tool
                tool_info = next(tool for tool in mcp_client_manager.tools if tool["name"] == tool_name)
                server_name = tool_info["server"]

                # Execute tool call
                tool_result = await mcp_client_manager.execute_tool(
                    server_name=server_name,
                    tool_name=tool_name, 
                    arguments=tool_args)

                responses.append(f"[Calling tool {tool_name} with args {tool_args}]")

                tool_result_msg = llm_client.parse_tool_result(tool_call=tool_call, tool_result=tool_result)
                messages.append(tool_result_msg)

        return {
            "responses": responses,
            "messages": messages
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)