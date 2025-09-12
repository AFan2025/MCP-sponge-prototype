"""
MCP AI Agent Client
===================

This is the main AI agent that connects to multiple MCP servers using the official MCP library.
The agent can communicate with:
1. Custom local MCP server (local_server.py) - for custom business tools
2. External MCP servers (OpenWeather, Google Search) - for standard services

Key Architecture:
- Uses real MCP protocol for inter-process communication
- Supports multiple simultaneous server connections
- Provides natural language interface for tool selection
- Extensible design for adding more servers/tools

Author: Assistant
Date: August 15, 2025
"""

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

# MCP Library imports - the official MCP client components
try:
    from mcp import ClientSession, stdio_client, StdioServerParameters
    from mcp.types import Tool, CallToolResult
except ImportError:
    print("❌ MCP library not installed. Install with: pip install mcp")
    print("📚 For development, you can also run: pip install -e .")
    sys.exit(1)

# Set up comprehensive logging to track MCP communications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/alex/Desktop/projectCompliance/Llama-3.1-8B-Instruct-hf"  # Default model path, can be overridden

@dataclass
class MCPServerConfig:
    """
    Configuration for MCP server connections.
    This defines how to connect to each server type.
    """
    name: str                    # Unique identifier for the server
    command: List[str]           # Command to start the server process
    description: str             # Human-readable description
    env_vars: Dict[str, str]     # Environment variables needed
    enabled: bool = True         # Whether to connect to this server
    cwd: str = "."                # Working directory for the server
    args: List[str] = None              # Arguments for the server process

class MCPAIAgent:
    """
    Advanced AI Agent using real MCP protocol to communicate with multiple servers.
    
    This agent demonstrates a production-ready architecture where:
    1. Each tool category runs in its own isolated server process
    2. Communication happens via standardized MCP protocol
    3. Tools can be written in any language (Python, Node.js, Go, etc.)
    4. Easy to scale, secure, and maintain
    
    Key Features:
    - Multi-server connection management
    - Automatic tool discovery from all connected servers
    - Natural language intent parsing for tool selection
    - Robust error handling and connection recovery
    - Extensible architecture for adding new server types
    """
    
    def __init__(self, use_local_llm = True, model_name = MODEL_PATH):
        """Initialize the MCP AI Agent with empty server registry."""
        self.server_configs = {}     # Registry of server configurations
        self.active_sessions = {}    # Active MCP client sessions
        self.available_tools = []    # Aggregated tools from all servers
        self.conversation_history = []  # Chat history for context

        self.use_local_llm = use_local_llm  # Whether to use local LLM or remote
        self.model_name = model_name  # Name/Path to the LLM model
        self.model = None
        self.tokenizer = None
        self.device = None

        ##for timing
        self.start_time = None
        self.end_time = None

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        if use_local_llm:
            self._load_local_model()
        
        logger.info("🤖 MCP AI Agent initialized")
        
        # Load default server configurations
        self._setup_default_servers()

    def _load_local_model(self):
        """
        Load local LLM model using HF library.
        """
        logger.info(f"🔍 Loading local model: {self.model_name}")
        
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Using device: {self.device}")

        torch.cuda.empty_cache()  # Clear CUDA cache for fresh start
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                # load_in_8bit=True,  # Use 8-bit quantization for efficiency
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",  # Ensure left padding for compatibility
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ Local model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            sys.exit(1)
    
    def _setup_default_servers(self):
        """
        Configure the default set of MCP servers to connect to.
        
        CRITICAL: This is where you define which servers your agent will use.
        Each server runs as a separate process and communicates via MCP protocol.
        """
        
        # 1. CUSTOM LOCAL SERVER - Your business-specific tools
        # This server contains tools you've created for your specific use case
        self.server_configs["local_custom"] = MCPServerConfig(
            name="local_custom",
            # command=["python", "local_server.py"],  # Start our custom server
            command=["python"],
            description="Custom business tools and local functionality",
            env_vars={},  # No special environment needed for local tools
            # cwd=".",
            args=["local_server.py"]
        )
        
        # 2. OPENWEATHER SERVER - Weather information services
        # External server that wraps OpenWeatherMap API
        self.server_configs["openweather"] = MCPServerConfig(
            name="openweather",
            command=["python", "-m", "mcp_servers.weather", "--api-key", "YOUR_OPENWEATHER_KEY"],
            description="Weather data from OpenWeatherMap API",
            env_vars={"OPENWEATHER_API_KEY": "your-api-key-here"},
            enabled=False  # Disabled by default - requires API key setup
        )
        
        # 3. GOOGLE SEARCH SERVER - Web search capabilities
        # External server for web search functionality
        self.server_configs["google_search"] = MCPServerConfig(
            name="google_search", 
            command=["python", "-m", "mcp_servers.search", "--api-key", "YOUR_GOOGLE_KEY"],
            description="Web search via Google Custom Search API",
            env_vars={"GOOGLE_API_KEY": "your-google-api-key"},
            enabled=False  # Disabled by default - requires API key setup
        )
        
        # 4. FILESYSTEM SERVER - File operations (example of Node.js server)
        # This demonstrates connecting to servers written in other languages
        self.server_configs["filesystem"] = MCPServerConfig(
            name="filesystem",
            command=["node", "filesystem-mcp-server.js"],
            description="File system operations and text file management",
            env_vars={},
            enabled=False  # Optional - enable if you have the Node.js server
        )
        
        logger.info(f"📋 Configured {len(self.server_configs)} potential MCP servers")
    
    async def connect_to_servers(self):
        """
        Establish connections to all enabled MCP servers.
        
        CRITICAL: This is where the actual MCP protocol connections are made.
        Each connection involves:
        1. Starting the server process
        2. Establishing stdio communication channel
        3. Performing MCP handshake
        4. Discovering available tools
        """
        
        logger.info("🔌 Starting MCP server connections...")
        
        connection_tasks = []
        for config in self.server_configs.values():
            if config.enabled:
                connection_tasks.append(self._connect_single_server(config))
        
        # Connect to all servers concurrently
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Report connection results
        successful_connections = 0
        for i, result in enumerate(results):
            config_name = list(self.server_configs.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to connect to {config_name}: {result}")
            else:
                successful_connections += 1
                logger.info(f"✅ Connected to {config_name}")
        
        logger.info(f"📡 MCP Agent ready with {successful_connections} active server connections")
        
        # Discover all available tools from connected servers
        await self._discover_all_tools()
    
    async def _connect_single_server(self, config: MCPServerConfig):
        """
        Connect to a single MCP server using the official MCP client.
        
        Args:
            config: Server configuration with connection details
            
        Raises:
            Exception: If connection fails for any reason
        """
        try:
            logger.info(f"🔗 Connecting to MCP server: {config.name}")
            
            # Create server parameters for stdio connection
            server_params = StdioServerParameters(
                command=" ".join(config.command),
                cwd=config.cwd,
                args=config.args,
                env=config.env_vars
            )
            
            # # Create stdio client and session - this starts the server process
            # # The server process will communicate via stdin/stdout
            # stdio_conn = stdio_client(server_params)
            # read_stream, write_stream = await stdio_conn.__aenter__()
            
            # # Create MCP client session
            # session = ClientSession(read_stream, write_stream)

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            # Initialize the session - this performs the MCP handshake
            await self.session.initialize()

            # Store the active session and connection for later use
            self.active_sessions[config.name] = {
                # 'session': session,
                # 'connection': stdio_conn
                'session': self.session,
                'connection': stdio_transport
            }
            
            logger.info(f"✅ Successfully connected to {config.name}")
            
        except Exception as e:
            logger.error(f"❌ Connection failed for {config.name}: {str(e)}")
            raise
    
    async def _discover_all_tools(self):
        """
        Discover tools from all connected MCP servers.
        
        CRITICAL: This aggregates tools from multiple servers into a unified catalog.
        The agent will know about all available tools regardless of which server provides them.
        """
        
        self.available_tools = []
        
        for server_name, session_info in self.active_sessions.items():
            try:
                logger.info(f"🔍 Discovering tools from {server_name}...")
                
                # Get the session from the session info
                session = session_info['session']
                
                # Call MCP list_tools to get available tools
                tools_response = await session.list_tools()
                
                # Add server information to each tool for routing
                for tool in tools_response.tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                        "server": server_name,  # Track which server provides this tool
                        "server_description": self.server_configs[server_name].description
                    }
                    self.available_tools.append(tool_info)
                
                logger.info(f"📦 Found {len(tools_response.tools)} tools from {server_name}")
                
            except Exception as e:
                logger.error(f"❌ Tool discovery failed for {server_name}: {e}")
        
        total_tools = len(self.available_tools)
        logger.info(f"🎯 Total available tools: {total_tools}")
        
        # Log available tools for debugging
        for tool in self.available_tools:
            logger.debug(f"  📋 {tool['name']} (from {tool['server']}): {tool['description']}")
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available tools from all connected servers.
        
        Returns:
            List of tool descriptions with server attribution
        """
        return self.available_tools.copy()

    def _create_tool_calling_prompt(self, user_message: str) -> str:
        """
        Create optimized prompt for Llama-3.1-8B-Instruct tool calling.
        
        Llama-3.1 works best with specific formatting for tool calls.
        """
        
        # Build comprehensive tool documentation
        tools_doc = "AVAILABLE TOOLS:\n\n"
        for i, tool in enumerate(self.available_tools, 1):
            tools_doc += f"{i}. **{tool['name']}** (Server: {tool['server']})\n"
            tools_doc += f"   Description: {tool['description']}\n"
            
            # Add parameter details for better tool understanding
            if 'input_schema' in tool and 'properties' in tool['input_schema']:
                tools_doc += "   Parameters:\n"
                for param, details in tool['input_schema']['properties'].items():
                    param_type = details.get('type', 'string')
                    param_desc = details.get('description', 'No description')
                    required = param in tool['input_schema'].get('required', [])
                    req_marker = " (required)" if required else " (optional)"
                    tools_doc += f"     - {param} ({param_type}){req_marker}: {param_desc}\n"
            tools_doc += "\n"
        
        # Llama-3.1 optimized system prompt
        system_prompt = f"""You are an intelligent AI assistant with access to external tools via MCP protocol.

{tools_doc}

TOOL CALLING INSTRUCTIONS:
1. Analyze the user's request carefully
2. If a tool can help, respond with this EXACT JSON format:
   {{"tool_call": {{"name": "exact_tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}}}

3. If no tool is needed, respond with:
   {{"response": "your helpful conversational response"}}

4. Attempt to use the tool main_analysis for most requests, unless another tool is more appropriate.

5. Give only one tool call or response at a time.

EXAMPLES:

User: "What's the weather in Paris?"
Assistant: {{"tool_call": {{"name": "get_weather", "arguments": {{"location": "Paris", "units": "celsius"}}}}}}

User: "Calculate 15 * 23 + 7"
Assistant: {{"tool_call": {{"name": "calculate", "arguments": {{"expression": "15 * 23 + 7"}}}}}}

User: "Hello, how are you?"
Assistant: {{"response": "Hello! I'm doing well. I'm an AI assistant with access to various tools including weather, calculations, search, and more. How can I help you today?"}}

IMPORTANT:
- Always use exact tool names from the list above
- Extract parameters accurately from user messages
- Respond ONLY with valid JSON
- Be precise and helpful"""

        # Format using Llama-3.1 chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Use Llama-3.1's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback manual formatting for Llama-3.1
            formatted_prompt = "<|begin_of_text|>"
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                elif msg["role"] == "user":
                    formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted_prompt
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the appropriate MCP server.
        
        This method:
        1. Finds which server provides the requested tool
        2. Routes the call to that specific server
        3. Returns the result in a standardized format
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            Standardized result dictionary with success status and data
        """
        
        logger.info(f"🛠️ Tool call requested: {tool_name} with args: {arguments}")
        
        # Find which server provides this tool
        target_server = None
        target_tool = None
        
        for tool in self.available_tools:
            if tool["name"] == tool_name:
                target_server = tool["server"]
                target_tool = tool
                break
        
        if not target_server:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {[t['name'] for t in self.available_tools]}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "available_tools": [t['name'] for t in self.available_tools]
            }
        
        # Execute the tool on the appropriate server
        try:
            session = self.active_sessions[target_server]['session']
            
            logger.info(f"⚡ Executing {tool_name} on server {target_server}")
            
            # Make the actual MCP call_tool request
            result = await session.call_tool(tool_name, arguments)
            
            logger.info(f"✅ Tool {tool_name} executed successfully")
            
            return {
                "success": True,
                "tool_name": tool_name,
                "server": target_server,
                "arguments": arguments,
                "result": result.content,  # MCP result content
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "tool_name": tool_name,
                "server": target_server,
                "arguments": arguments,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    async def parse_user_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Parse user message to determine intent and extract parameters.
        
        CRITICAL: This is where natural language gets converted to tool calls.
        In production, you'd use an LLM here, but this shows the concept with patterns.
        
        Args:
            user_message: Natural language request from user
            
        Returns:
            Dict with tool_name and arguments, or None if no tool needed
        """
        if self.use_local_llm or self.model:
            return await self._llm_parse_intent(user_message)

        message_lower = user_message.lower().strip()
        logger.info(f"🧠 Parsing intent from: '{user_message}'")
        
        # WEATHER INTENT PATTERNS
        weather_patterns = [
            r"weather.*in\s+(\w+)",
            r"temperature.*in\s+(\w+)", 
            r"forecast.*for\s+(\w+)",
            r"how.*weather.*(\w+)"
        ]
        
        for pattern in weather_patterns:
            match = re.search(pattern, message_lower)
            if match:
                location = match.group(1).title()
                units = "fahrenheit" if "fahrenheit" in message_lower or "°f" in message_lower else "celsius"
                
                return {
                    "tool_name": "get_weather",  # Assumes weather tool is named this
                    "arguments": {"location": location, "units": units}
                }
        
        # SEARCH INTENT PATTERNS
        search_patterns = [
            r"search\s+for\s+(.+)",
            r"look\s+up\s+(.+)",
            r"find\s+information\s+about\s+(.+)",
            r"google\s+(.+)"
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, message_lower)
            if match:
                query = match.group(1).strip()
                return {
                    "tool_name": "web_search",  # Assumes search tool is named this
                    "arguments": {"query": query, "max_results": 5}
                }
        
        # MATH/CALCULATION PATTERNS
        math_patterns = [
            r"calculate\s+(.+)",
            r"compute\s+(.+)",
            r"solve\s+(.+)",
            r"what\s+is\s+([0-9+\-*/().\s]+)"
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, message_lower)
            if match:
                expression = match.group(1).strip()
                return {
                    "tool_name": "calculate",  # Assumes calculator tool exists
                    "arguments": {"expression": expression}
                }
        
        # FILE OPERATIONS PATTERNS
        if "read file" in message_lower or "open file" in message_lower:
            # Extract file path (basic pattern)
            path_match = re.search(r"file\s+([/\w\-.]+)", message_lower)
            if path_match:
                file_path = path_match.group(1)
                return {
                    "tool_name": "read_file",
                    "arguments": {"path": file_path}
                }
        
        # CUSTOM TOOLS - Add patterns for your custom tools here
        # Example: Business-specific operations
        if "analyze sales" in message_lower:
            return {
                "tool_name": "analyze_sales_data", 
                "arguments": {"period": "monthly"}
            }
        
        if "generate report" in message_lower:
            return {
                "tool_name": "generate_report",
                "arguments": {"type": "summary"}
            }
        
        logger.info("❓ No specific tool intent detected")
        return None
    
    async def _llm_parse_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Using local model to parse user intent and select tools

        Outputs using standardized MCP format.
        """
        # if not self.use_local_llm or not self.model:
        #     logger.warning("⚠️ Local LLM parsing is disabled or model not loaded, using pattern matching")
        #     return self.parse_user_intent(user_message)
        try:
            prompt = self._create_tool_calling_prompt(user_message)

            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,  # Reasonable context window
            ).to(self.device) #possible error mode with bad devicing?

            print(inputs.input_ids.shape)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids, 
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,  # Limit to reasonable response length
                    temperature=0.1,
                    top_p=0.95,
                    top_k=50,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_strings=["<|eot_id|>", "<|end_of_text|>"],
                    tokenizer=self.tokenizer,  # Required for stop_strings functionality
                    early_stopping=True
                    )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            logger.info(f"🧠 Llama-3.1 raw response: {generated_text}")

            return self._parse_llm_tool_response(generated_text, user_message)
        
        except Exception as e:
            logger.error(f"❌ LLM intent parsing failed: {e}, using standard pattern matching")
            raise e
    
    def _parse_llm_tool_response(self, llm_response: str, original_message: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM-generated response to extract tool calls.
        
        Args:
            generated_text: Raw output from the LLM
            user_message: Original user input for context
        """
        try:
            cleaned_response = self._preprocess_tool_call(llm_response)

            parsed = json.loads(cleaned_response)
            # parsed = json.loads(llm_response.strip())

            if "tool_call" in parsed:
                tool_call = parsed["tool_call"]
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})

                logger.info(f"🔍 LLM parsed tool call: {tool_name} with args: {arguments}")

                return {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "reasoning": f"{self.model_name} decision",
                    "confidence": "high"
                }
            elif "response" in parsed:
                response_text = parsed["response"]
                logger.info(f"💬 LLM conversational response: {response_text}")
                return {
                    "tool_name": None,
                    "arguments": {},
                    "response": response_text,
                    "reasoning": f"{self.model_name} conversational response",
                    "confidence": "high"
                }
            
        except json.JSONDecodeError:
            logger.error(f"❌ Failed to decode LLM response as JSON: {llm_response}")
            raise ValueError("Invalid JSON format in LLM response")

        except Exception as e:
            logger.error(f"❌ Failed to parse LLM response: {e}")

        return None

    def _preprocess_tool_call(self, llm_response: str) -> str:
        """
        Preprocess LLM response to ensure valid JSON format.
        
        This handles common issues like unmatched braces, newlines, etc.
        """
        logger.info("🔧 Cleaning LLM response for tool call extraction")
        # Remove leading/trailing whitespace
        cleaned_response = llm_response.strip()

        #remove everything until the first valid JSON object
        json_start = re.search(r'\{', cleaned_response)
        if json_start:
            cleaned_response = cleaned_response[json_start.start():]

        #removes any trailing text after the first valid JSON object
        json_end = cleaned_response.rfind('}')
        if json_end != -1:
            cleaned_response = cleaned_response[:json_end + 1]
        
        # Remove code block markers if present
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        # Replace newlines with escaped characters
        cleaned_response = cleaned_response.replace('\n', '\\n').replace('\r', '\\r')

        logger.info(f"🔍 Cleaned response: {cleaned_response}")  # Log first 100 chars for brevity

        # Check and fix unmatched curly braces
        open_braces = cleaned_response.count('{')
        close_braces = cleaned_response.count('}')
        if open_braces > close_braces:
            # Add missing closing braces
            cleaned_response += '}' * (open_braces - close_braces)
            logger.info(f"🔧 Added {open_braces - close_braces} missing closing brace(s)")
        elif close_braces > open_braces:
            # Remove extra closing braces from the end
            extra_braces = close_braces - open_braces
            for _ in range(extra_braces):
                if cleaned_response.endswith('}'):
                    cleaned_response = cleaned_response[:-1]
            logger.info(f"🔧 Removed {extra_braces} extra closing brace(s)")
        
        return cleaned_response

    async def process_message(self, user_message: str) -> str:
        """
        Main message processing pipeline using local LLM.
        
        This orchestrates the full flow:
        1. Parse natural language for intent using LLM
        2. Execute tools if needed
        3. Allow LLM to process tool results and make additional calls
        4. Generate final response
        
        Args:
            user_message: User's natural language input
            
        Returns:
            Formatted response string
        """
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Multi-turn conversation with tool support
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        tool_results = []
        
        current_context = user_message
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Processing iteration {iteration}")
            logger.info(f"💬 Processing message: '{current_context}'")            
            
            # Parse intent using LLM
            intent = await self.parse_user_intent(current_context)
            
            if intent and intent.get("tool_name"):
                tool_name = intent["tool_name"]
                arguments = intent["arguments"]
                
                logger.info(f"🎯 Tool call detected: {tool_name}")
                
                # Execute the tool via MCP
                tool_result = await self.call_tool(tool_name, arguments)
                logger.info(f"🔧 Tool execution result: {tool_result}")
                tool_results.append(tool_result)
                
                if tool_result["success"]:
                    # Prepare context for next iteration with tool result
                    tool_output = self._extract_tool_content(tool_result["result"])
                    current_context = f"Tool {tool_name} returned: {tool_output}\n\nBased on this result, what should I do next? If no further action is needed, provide a final response to the user."
                    logger.info(f"✅ Tool executed successfully, continuing conversation")
                else:
                    # Tool failed, let LLM handle the error
                    error_msg = tool_result.get('error', 'Unknown error')
                    current_context = f"Tool {tool_name} failed with error: {error_msg}\n\nPlease provide an appropriate response to the user about this error."
                    logger.error(f"❌ Tool execution failed: {error_msg}")
            
            elif intent and intent.get("response"):
                # LLM provided a conversational response - we're done
                final_response = intent["response"]
                
                # Include tool results in context if any were executed
                if tool_results:
                    response_with_context = self._format_response_with_tools(final_response, tool_results)
                    final_response = response_with_context
                
                # Add response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response, 
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"✅ Message processed in {iteration} iteration(s)")
                return final_response
            
            else:
                # Fallback: generate conversational response
                logger.info("🤖 No tool call or response detected, generating conversational response")
                final_response = await self._generate_conversational_response(current_context)
                
                # Add response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response, 
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"✅ Message processed with fallback in {iteration} iteration(s)")
                return final_response
        
        # Max iterations reached
        logger.warning(f"⚠️ Max iterations ({max_iterations}) reached")
        fallback_response = "I've been working on your request but reached my processing limit. Let me provide what I have so far."
        
        if tool_results:
            fallback_response = self._format_response_with_tools(fallback_response, tool_results)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": fallback_response, 
            "timestamp": datetime.now().isoformat()
        })
        
        return fallback_response
    
    def _extract_tool_content(self, tool_result_content) -> str:
        """
        Extract readable content from tool result for LLM processing.
        
        Args:
            tool_result_content: Raw result content from MCP tool
            
        Returns:
            Formatted string content
        """
        if isinstance(tool_result_content, list):
            # MCP often returns list of content items
            formatted_content = ""
            for item in tool_result_content:
                if hasattr(item, "text"):
                    formatted_content += item.text
                elif item.get("type") == "text":
                    formatted_content += item.get("text", "")
                elif item.get("type") == "image":
                    formatted_content += f"[Image: {item.get('alt', 'No description')}]"
            return formatted_content
        else:
            return str(tool_result_content)
    
    def _format_response_with_tools(self, llm_response: str, tool_results: List[Dict[str, Any]]) -> str:
        """
        Format final response including tool execution context.
        
        Args:
            llm_response: Response from LLM
            tool_results: List of tool execution results
            
        Returns:
            Formatted response with tool context
        """
        if not tool_results:
            return llm_response
        
        # Add tool execution summary
        response = llm_response + "\n\n---\n"
        response += f"🔧 **Tool Execution Summary** ({len(tool_results)} tool(s) used):\n"
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get("tool_name", "unknown")
            server = result.get("server", "unknown")
            success = result.get("success", False)
            status = "✅ Success" if success else "❌ Failed"
            
            response += f"{i}. **{tool_name}** (via {server}): {status}\n"
        
        return response
    
    def _format_tool_response(self, tool_result: Dict[str, Any]) -> str:
        """
        Format tool execution results into user-friendly responses.
        
        Args:
            tool_result: Result dictionary from tool execution
            
        Returns:
            Formatted response string
        """
        
        tool_name = tool_result["tool_name"]
        server = tool_result["server"]
        result_content = tool_result["result"]
        
        # Handle different result content types from MCP
        if isinstance(result_content, list):
            # MCP often returns list of content items
            formatted_content = ""
            for item in result_content:
                if item.get("type") == "text":
                    formatted_content += item.get("text", "")
                elif item.get("type") == "image":
                    formatted_content += f"[Image: {item.get('alt', 'No description')}]"
        else:
            formatted_content = str(result_content)
        
        # Add server attribution and tool info
        response = f"🛠️ **Result from {tool_name}** (via {server}):\n\n"
        response += formatted_content
        
        return response
    
    async def _generate_conversational_response(self, user_message: str) -> str:
        """
        Generate conversational responses using the local LLM.
        
        Args:
            user_message: User's message or current context
            
        Returns:
            Conversational response from LLM
        """
        if not self.use_local_llm or not self.model:
            # Fallback to basic responses if LLM not available
            return self._generate_fallback_response(user_message)
        
        try:
            # Create a conversational prompt without tool calling
            conversation_prompt = self._create_conversational_prompt(user_message)
            
            inputs = self.tokenizer(
                conversation_prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=4096,  # Reasonable context window for conversation
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids, 
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=256,  # Shorter responses for conversation
                    temperature=0.3,  # More focused responses
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_strings=["<|eot_id|>", "<|end_of_text|>"],
                    tokenizer=self.tokenizer,
                    early_stopping=True,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            logger.info(f"🤖 Generated conversational response: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ LLM conversation generation failed: {e}")
            return self._generate_fallback_response(user_message)
    
    def _create_conversational_prompt(self, user_message: str) -> str:
        """
        Create a prompt optimized for conversational responses.
        
        Args:
            user_message: Current message or context
            
        Returns:
            Formatted prompt for conversation
        """
        # Build context from recent conversation history
        context_messages = []
        
        # Add recent conversation history (last 5 exchanges)
        recent_history = self.conversation_history[-10:] if self.conversation_history else []
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_messages.append({"role": role, "content": content})
        
        # Add current message
        context_messages.append({"role": "user", "content": user_message})
        
        # Create system prompt for conversational AI
        system_prompt = f"""You are an intelligent AI assistant with access to {len(self.available_tools)} tools across {len(self.active_sessions)} connected servers.

You can help with:
- Calculations and mathematical problems
- Data analysis and statistics
- Text analysis and summarization
- Business metrics and reporting
- General questions and conversation

Respond naturally and helpfully. Be concise but informative. If you think a tool might help with the user's request, suggest it, but focus on being conversational and helpful.

Available tools include: {', '.join([tool['name'] for tool in self.available_tools[:5]])}{'...' if len(self.available_tools) > 5 else ''}"""
        
        # Use Llama-3.1's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "system", "content": system_prompt}] + context_messages
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback manual formatting
            formatted_prompt = "<|begin_of_text|>"
            formatted_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            
            for msg in context_messages:
                role = msg["role"]
                content = msg["content"]
                formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            
            formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted_prompt
    
    def _generate_fallback_response(self, user_message: str) -> str:
        """
        Generate basic responses when LLM is not available.
        
        Args:
            user_message: User's message
            
        Returns:
            Basic conversational response
        """
        message_lower = user_message.lower()
        
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey"]):
            tool_count = len(self.available_tools)
            server_count = len(self.active_sessions)
            return (f"Hello! 👋 I'm an MCP-powered AI agent with access to {tool_count} tools "
                   f"across {server_count} connected servers. I can help with weather, search, "
                   f"calculations, file operations, and custom business tasks. What would you like to do?")
        
        elif "help" in message_lower or "what can you do" in message_lower:
            tools_by_server = {}
            for tool in self.available_tools:
                server = tool["server"]
                if server not in tools_by_server:
                    tools_by_server[server] = []
                tools_by_server[server].append(f"• {tool['name']}: {tool['description']}")
            
            response = "🔧 **Available Tools by Server:**\n\n"
            for server, tools in tools_by_server.items():
                response += f"**{server}** ({self.server_configs[server].description}):\n"
                response += "\n".join(tools) + "\n\n"
            
            return response
        
        elif "servers" in message_lower:
            response = "📡 **Connected MCP Servers:**\n\n"
            for server_name, config in self.server_configs.items():
                status = "🟢 Connected" if server_name in self.active_sessions else "🔴 Disconnected"
                response += f"• **{server_name}**: {config.description} - {status}\n"
            
            return response
        
        else:
            return ("I'm not sure how to help with that. Try asking me to:\n"
                   "• Check weather: 'What's the weather in Paris?'\n" 
                   "• Search web: 'Search for Python tutorials'\n"
                   "• Calculate: 'Calculate 15 * 23'\n"
                   "• Type 'help' to see all available tools")
    
    async def shutdown(self):
        """
        Cleanly shutdown all MCP server connections.
        
        CRITICAL: Proper cleanup prevents zombie processes and resource leaks.
        """
        
        logger.info("🔌 Shutting down MCP connections...")
        
        shutdown_tasks = []
        for server_name, session_info in self.active_sessions.items():
            logger.info(f"🔚 Closing connection to {server_name}")
            # Close both the session and the stdio connection
            shutdown_tasks.append(session_info['session'].close())
            shutdown_tasks.append(session_info['connection'].__aexit__(None, None, None))
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.active_sessions.clear()
        logger.info("✅ All MCP connections closed")

# ==============================================================================
# DEMO AND TESTING FUNCTIONS
# ==============================================================================

async def run_interactive_demo():
    """
    Interactive demo mode for testing the MCP AI Agent.
    This provides a command-line interface to test all functionality.
    """
    
    print("🤖 MCP AI Agent - Interactive Demo")
    print("=" * 50)
    print("This agent connects to multiple MCP servers and provides unified access to tools.")
    print("Commands:")
    print("  • Type natural language requests")
    print("  • 'tools' - List all available tools")
    print("  • 'servers' - Show server status") 
    print("  • 'quit' - Exit the demo")
    print("=" * 50)
    
    # Initialize the agent
    agent = MCPAIAgent(
        use_local_llm=True,  # Set to False to disable local LLM usage
        model_name=MODEL_PATH  # Path to your local LLM model
    )
    
    try:
        # Connect to all configured servers
        await agent.connect_to_servers()
        
        if not agent.active_sessions:
            print("❌ No MCP servers connected. Check your configuration.")
            print("💡 Make sure local_server.py exists and is executable")
            return
        
        print(f"\n🚀 Agent ready with {len(agent.active_sessions)} active server connections!")
        print(f"📦 Total tools available: {len(agent.available_tools)}")
        print()
        
        # Interactive loop
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'tools':
                    tools = agent.list_available_tools()
                    print(f"\n🔧 Available Tools ({len(tools)}):")
                    for tool in tools:
                        print(f"  • {tool['name']} (from {tool['server']}): {tool['description']}")
                    print()
                    continue
                
                if user_input.lower() == 'servers':
                    print("\n📡 Server Status:")
                    for name, config in agent.server_configs.items():
                        status = "🟢 Connected" if name in agent.active_sessions else "🔴 Disconnected"
                        print(f"  • {name}: {status} - {config.description}")
                    print()
                    continue
                
                # Process the message
                print("🤖 Agent: ", end="", flush=True)
                response = await agent.process_message(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
                
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"\n❌ An error occurred: {e}")
        
    finally:
        # Clean shutdown
        await agent.shutdown()
        print("👋 MCP AI Agent demo ended. Goodbye!")

async def run_automated_demo():
    """
    Automated demo showing various agent capabilities.
    """
    
    print("🤖 MCP AI Agent - Automated Demo")
    print("=" * 50)
    
    agent = MCPAIAgent(
        use_local_llm=True,  # Set to False to disable local LLM usage
        model_name=MODEL_PATH  # Path to your local LLM model
    )
    
    try:
        await agent.connect_to_servers()
        print("FULLY CONNECTED")
        
        if not agent.active_sessions:
            print("❌ No servers connected - check configuration")
            return
        
        # Test messages demonstrating different capabilities
        test_messages = [
            "Hello! What can you help me with?",
            "What tools do you have available?", 
            "Calculate 25 * 17 + sqrt(144)",
            # "What's the weather like in London?",
            # "Search for machine learning tutorials",
            # "Generate a sales report",
            "Please analyze the sales data for last quarter",
            "Thanks for the demonstration!"
        ]
        
        print(f"📡 Connected to {len(agent.active_sessions)} servers")
        print(f"🔧 Available tools: {len(agent.available_tools)}")
        print("\n" + "=" * 50)
        print("🎭 Automated Conversation Demo")
        print("=" * 50)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test {i} ---")
            print(f"👤 User: {message}")
            
            response = await agent.process_message(message)
            print(f"🤖 Agent: {response}")
            
            # Brief pause between messages
            await asyncio.sleep(1)
        
        print("\n" + "=" * 50)
        print("✅ Automated demo completed successfully!")
        
    finally:
        await agent.shutdown()

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    """
    Main entry point - supports both interactive and automated demo modes.
    """
    
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        print("Starting automated demo...")
        asyncio.run(run_automated_demo())
    else:
        print("Starting interactive demo...")
        asyncio.run(run_interactive_demo())
