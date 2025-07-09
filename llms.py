from abc import ABC, abstractmethod
from anthropic import Anthropic
import openai
from pydantic import BaseModel
from typing import Dict, List, Optional
from google import genai

import json

class LLMClient(ABC):
    class ToolCall(BaseModel):
        id: str
        name: str
        input: dict

    class Response(BaseModel):
        content: Dict
        text_content: Optional[List[str]] = []
        tool_calls: List["LLMClient.ToolCall"] = []

    @abstractmethod
    async def create_message(self, messages: List[Dict]) -> Dict:
        pass

    @abstractmethod
    def parse_response(self, response) -> Response:
        pass

    @abstractmethod
    def parse_tool_result(self, tool_call, tool_result) -> Dict:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, tools):
        openai.api_key = api_key
        self.tools = tools

    def _prepare_tool_obj(self, tools):
        """
        Remove the 'default' key from input_schema['properties'].<property_name> in each object.
        
        :param json_data: List of JSON objects
        :return: Updated list with 'default' keys removed
        """
        for obj in tools:
            if "input_schema" in obj and "properties" in obj["input_schema"]:
                for prop in obj["input_schema"]["properties"].values():
                    prop.pop("default", None)
        return tools


    async def create_message(self, messages: List[Dict]) -> Dict:
        preprocessed_tools = self._prepare_tool_obj(self.tools)
        print(preprocessed_tools)
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": f"[{tool['server']}] {tool['description']}",
                    "parameters": tool["input_schema"],
                    "strict": True
                }
            }
            for tool in preprocessed_tools
        ]

        return openai.ChatCompletion.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages,
            tools=openai_tools
        )

    def parse_response(self, response) -> LLMClient.Response:
        text_content = []
        tool_calls = []

        if response.choices[0].message.content: 
            text_content.append(response.choices[0].message.content)

        if response.choices[0].message.tool_calls:
            for content in response.choices[0].message.tool_calls:
                tool_calls.append(LLMClient.ToolCall(id=content.id, name=content.function.name, input=json.loads(content.function.arguments)))

        parsed_response = LLMClient.Response(
            content=response.choices[0].message.to_dict(),
            text_content=text_content,
            tool_calls=tool_calls
        )
        return parsed_response
    
    def parse_tool_result(self, tool_call, tool_result) -> Dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result.content
        } 
    
class AnthropicClient(LLMClient):

    def __init__(self, api_key: str, tools):
        self.client = Anthropic(api_key=api_key)
        self.tools = tools

    async def create_message(self, messages: List[Dict]) -> Dict:
        claude_tools = [
            {
                "name": tool["name"],
                "description": f"[{tool['server']}] {tool['description']}",
                "input_schema": tool["input_schema"]
            }
            for tool in self.tools
        ]
        
        return self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=claude_tools
        )

    def parse_response(self, response) -> LLMClient.Response:
        text_content = []
        tool_calls = []
        for content in response.content: 
            if content.type == 'text':
                text_content.append(content.text)
            elif content.type == 'tool_use':
                tool_calls.append(LLMClient.ToolCall(id=content.id, name=content.name, input=content.input))

        parsed_response = LLMClient.Response(
            content={
                "role": "assistant",
                "content": response.content
            },
            text_content=text_content,
            tool_calls=tool_calls
        )
        return parsed_response
    
    def parse_tool_result(self, tool_call, tool_result) -> Dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_result.content
                }
            ]
        } 


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, function_declarations):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.function_declarations = function_declarations

    async def create_message(self, messages: list) -> dict:
        from google.genai import types
        # Wrap the function declarations into a Tool and config as per new Gemini API
        tools = types.Tool(function_declarations=self.function_declarations)
        config = types.GenerateContentConfig(tools=[tools])
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=config,
        )
        return response

    def parse_response(self, response) -> LLMClient.Response:
        text_content = []
        tool_calls = []
        # Gemini v2 function call responses
        candidates = getattr(response, "candidates", None)
        if candidates and hasattr(candidates[0].content, "parts"):
            parts = candidates[0].content.parts
            for part in parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(
                        LLMClient.ToolCall(
                            id=getattr(part.function_call, "name", ""),
                            name=getattr(part.function_call, "name", ""),
                            input=getattr(part.function_call, "args", {})
                        )
                    )
                elif hasattr(part, "text"):
                    text_content.append(part.text)
        else:
            # fallback: try to extract text
            if hasattr(response, "text"):
                text_content.append(response.text)

        parsed_response = LLMClient.Response(
            content={
                "role": "assistant",
                "content": getattr(response, "text", "") or getattr(response, "candidates", "")
            },
            text_content=text_content,
            tool_calls=tool_calls
        )
        return parsed_response

    def parse_tool_result(self, tool_call, tool_result) -> Dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_result.content
                }
            ]
        }