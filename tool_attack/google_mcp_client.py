import asyncio
import json
import logging
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
from datetime import datetime
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import AsyncExitStack

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.types import Tool, CallToolResult, ListToolsResult, TextContent
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP library not installed. Install with: pip install mcp")
    print("📚 For development, you can also run: pip install -e .")
    sys.exit(1)

class ManualMCPClient:
    def __init__(self, transport = "stdio"):
        self.transport = transport
        self.logger = logging.getLogger("ManualMCPClient")
        self.configs = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}

    async def connect_server_stdio(self, server_id: str, server_params: StdioServerParameters) -> Optional[ClientSession]:
            """Connect to an MCP server through stdio."""

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            # Initialize the session - this performs the MCP handshake
            await self.session.initialize()

            return self.session


    async def connect_server_sse(self, server_url: str, server_id: str) -> Optional[ClientSession]:
        """Connect to an MCP server based on its configuration."""

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session

        return session

if __name__ == "__main__":
    client = ManualMCPClient()



