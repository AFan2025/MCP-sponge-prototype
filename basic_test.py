"""
Basic AI Agent System using MCP (Model Context Protocol) Server
This example demonstrates how to create an AI agent that can use custom tools
through the MCP protocol to interact with external services.

Author: Assistant
Date: August 13, 2025
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import re
from datetime import datetime

# Set up logging to track what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: Define Custom Tools/Functions
# These are the actual implementations that our agent can call
# ============================================================================

def find_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    """
    Custom tool to find weather information for a given location.
    In a real implementation, this would call a weather API like OpenWeatherMap.
    
    Args:
        location: City name or coordinates (e.g., "New York", "London")
        units: Temperature units ("celsius" or "fahrenheit")
    
    Returns:
        Dictionary with weather information
    """
    logger.info(f"🌤️ Weather lookup requested for: {location} in {units}")
    
    # Mock weather data - in production, replace with real API call to OpenWeatherMap, etc.
    mock_weather_database = {
        "new york": {"temp_c": 22, "temp_f": 72, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp_c": 15, "temp_f": 59, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp_c": 28, "temp_f": 82, "condition": "Sunny", "humidity": 55},
        "paris": {"temp_c": 18, "temp_f": 64, "condition": "Overcast", "humidity": 70},
        "sydney": {"temp_c": 25, "temp_f": 77, "condition": "Clear", "humidity": 60}
    }
    
    # Look up weather data (case-insensitive)
    location_key = location.lower().strip()
    base_data = mock_weather_database.get(location_key, 
        {"temp_c": 20, "temp_f": 68, "condition": "Unknown", "humidity": 50})
    
    # Format response based on requested units
    temperature = base_data["temp_c"] if units == "celsius" else base_data["temp_f"]
    
    weather_response = {
        "location": location.title(),
        "temperature": temperature,
        "units": units,
        "condition": base_data["condition"],
        "humidity": base_data["humidity"],
        "wind_speed": 10,  # Mock wind speed
        "timestamp": datetime.now().isoformat(),
        "forecast": [
            {"day": "Today", "high": temperature + 3, "low": temperature - 5, "condition": base_data["condition"]},
            {"day": "Tomorrow", "high": temperature + 1, "low": temperature - 7, "condition": "Partly cloudy"},
            {"day": "Day After", "high": temperature - 2, "low": temperature - 8, "condition": "Cloudy"}
        ]
    }
    
    logger.info(f"✅ Weather data retrieved for {location}")
    return weather_response

def calculate_math(expression: str) -> Dict[str, Any]:
    """
    Custom tool to perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression as string (e.g., "2 + 2", "sqrt(16)", "sin(30)")
    
    Returns:
        Dictionary with calculation result or error information
    """
    logger.info(f"🧮 Math calculation requested: {expression}")
    
    try:
        # Define safe mathematical functions and constants
        # This prevents dangerous code execution while allowing useful math
        safe_math_functions = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, 
            "sqrt": lambda x: x**0.5,
            "sin": lambda x: __import__('math').sin(__import__('math').radians(x)),
            "cos": lambda x: __import__('math').cos(__import__('math').radians(x)),
            "tan": lambda x: __import__('math').tan(__import__('math').radians(x)),
            "log": lambda x: __import__('math').log(x),
            "pi": 3.14159265359, 
            "e": 2.71828182846
        }
        
        # Clean and prepare the expression
        clean_expression = expression.replace("^", "**")  # Convert ^ to ** for Python
        clean_expression = clean_expression.replace(" ", "")  # Remove spaces
        
        # Evaluate the expression safely
        result = eval(clean_expression, {"__builtins__": {}}, safe_math_functions)
        
        logger.info(f"✅ Math calculation successful: {expression} = {result}")
        return {
            "expression": expression,
            "result": result,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Math calculation failed: {expression} - Error: {str(e)}")
        return {
            "expression": expression,
            "error": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat()
        }

def get_system_info() -> Dict[str, Any]:
    """
    Custom tool to get basic system information.
    
    Returns:
        Dictionary with system information including platform, memory, CPU, etc.
    """
    logger.info("💻 System information requested")
    
    try:
        import platform
        import psutil
        
        system_data = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        logger.info("✅ System information retrieved successfully")
        return system_data
        
    except ImportError:
        logger.warning("⚠️ psutil not available, returning basic info only")
        import platform
        return {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "note": "Install 'psutil' for detailed system information"
        }

def search_web(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Custom tool to simulate web search functionality.
    In a real implementation, this would use a search API like Google Custom Search.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        Dictionary with search results
    """
    logger.info(f"🔍 Web search requested for: {query}")
    
    # Mock search results - replace with real search API in production
    mock_search_results = [
        {
            "title": f"Result 1 for '{query}'",
            "url": f"https://example.com/search1?q={query.replace(' ', '+')}",
            "snippet": f"This is a mock search result for the query '{query}'. In a real implementation, this would come from a search engine API."
        },
        {
            "title": f"Result 2 for '{query}'",
            "url": f"https://example.org/page2?search={query.replace(' ', '+')}",
            "snippet": f"Another mock result related to '{query}'. This demonstrates how search results would be formatted and returned."
        },
        {
            "title": f"Result 3 for '{query}'",
            "url": f"https://example.net/info3?q={query.replace(' ', '+')}",
            "snippet": f"Third mock result for '{query}'. Real implementation would use Google Search API, Bing API, or similar service."
        }
    ]
    
    # Limit results to max_results
    limited_results = mock_search_results[:max_results]
    
    search_response = {
        "query": query,
        "results_count": len(limited_results),
        "max_results": max_results,
        "results": limited_results,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Web search completed for: {query}")
    return search_response

# ============================================================================
# STEP 2: Define MCP Tool Structure
# This represents how tools are described in the MCP protocol
# ============================================================================

@dataclass
class MCPTool:
    """
    Represents a tool that can be called through the MCP protocol.
    This is the bridge between our Python functions and the MCP interface.
    Each tool has a name, description, parameter schema, and the actual function.
    """
    name: str                    # Unique identifier for the tool
    description: str             # Human-readable description of what the tool does
    parameters: Dict[str, Any]   # JSON Schema describing the tool's parameters
    function: callable           # The actual Python function to execute

# ============================================================================
# STEP 3: MCP Server Implementation
# This manages tool registration and execution
# ============================================================================

class MCPServer:
    """
    Mock MCP Server implementation that manages tools and handles execution.
    In a real MCP setup, this would communicate with an actual MCP server process,
    but for this example, we'll simulate the MCP protocol locally.
    
    The MCP Server is responsible for:
    1. Registering available tools
    2. Providing tool discovery (listing available tools)
    3. Executing tools with provided arguments
    4. Handling errors and returning results
    """
    
    def __init__(self):
        """Initialize the MCP server with an empty tool registry."""
        self.tools = {}  # Dictionary to store registered tools
        logger.info("🚀 MCP Server initializing...")
        self._register_default_tools()
        logger.info(f"📡 MCP Server ready with {len(self.tools)} tools")
    
    def _register_default_tools(self):
        """
        Register our custom tools with the MCP server.
        Each tool is defined with its schema following JSON Schema format.
        """
        
        # Register weather lookup tool
        weather_tool = MCPTool(
            name="find_weather",
            description="Get current weather information and forecast for a specific location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location to get weather for (e.g., 'New York', 'London')"
                    },
                    "units": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                        "description": "Temperature units for the response"
                    }
                },
                "required": ["location"]  # location is required, units is optional
            },
            function=find_weather
        )
        self.tools["find_weather"] = weather_tool
        logger.info("🌤️ Registered weather tool")
        
        # Register mathematical calculator tool
        math_tool = MCPTool(
            name="calculate_math",
            description="Perform mathematical calculations and evaluations safely",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(30)')"
                    }
                },
                "required": ["expression"]
            },
            function=calculate_math
        )
        self.tools["calculate_math"] = math_tool
        logger.info("🧮 Registered math calculator tool")
        
        # Register system information tool
        system_tool = MCPTool(
            name="get_system_info", 
            description="Get current system information including platform, memory, CPU, and disk usage",
            parameters={
                "type": "object",
                "properties": {},  # No parameters needed
                "required": []
            },
            function=get_system_info
        )
        self.tools["get_system_info"] = system_tool
        logger.info("💻 Registered system info tool")
        
        # Register web search tool
        search_tool = MCPTool(
            name="search_web",
            description="Search the web for information on a given topic",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to look up"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of search results to return"
                    }
                },
                "required": ["query"]
            },
            function=search_web
        )
        self.tools["search_web"] = search_tool
        logger.info("🔍 Registered web search tool")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Return a list of all available tools with their descriptions.
        This is what the AI agent calls to discover what tools are available.
        
        Returns:
            List of tool descriptions in JSON format
        """
        tool_list = []
        for tool in self.tools.values():
            tool_list.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        
        logger.info(f"📋 Tool list requested - returning {len(tool_list)} tools")
        return tool_list
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given arguments.
        This is the core of the MCP protocol - calling tools and returning results.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            Dictionary with tool execution result, success status, and any errors
        """
        logger.info(f"🔧 Tool execution requested: {tool_name}")
        
        # Check if the requested tool exists
        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "available_tools": list(self.tools.keys())
            }
        
        try:
            tool = self.tools[tool_name]
            logger.info(f"⚙️ Executing {tool_name} with arguments: {arguments}")
            
            # Call the actual function with the provided arguments
            # The ** operator unpacks the arguments dictionary
            result = tool.function(**arguments)
            
            logger.info(f"✅ Tool {tool_name} executed successfully")
            return {
                "success": True,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "tool_name": tool_name,
                "arguments": arguments,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

# ============================================================================
# STEP 4: AI Agent Implementation
# This processes natural language and decides which tools to use
# ============================================================================

class AIAgent:
    """
    The AI Agent that can understand user requests and decide which tools to use.
    
    In a real implementation, this would use a language model like GPT-4, Claude,
    or another LLM to understand natural language and make tool selection decisions.
    For this example, we'll use simple keyword-based intent detection.
    
    The agent is responsible for:
    1. Understanding user intent from natural language
    2. Selecting appropriate tools based on the request
    3. Extracting parameters from user messages
    4. Formatting tool responses into human-readable text
    5. Maintaining conversation context
    """
    
    def __init__(self, mcp_server: MCPServer):
        """
        Initialize the AI agent with access to the MCP server.
        
        Args:
            mcp_server: The MCP server instance that manages tools
        """
        self.mcp_server = mcp_server
        self.conversation_history = []  # Store conversation for context
        logger.info("🧠 AI Agent initialized")
    
    def _parse_user_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Parse user message to determine intent and extract parameters.
        
        In a production system, this would use:
        - Natural Language Processing (NLP)
        - Large Language Model (LLM) like GPT-4
        - Named Entity Recognition (NER)
        - Intent classification models
        
        For this example, we use simple keyword matching and pattern recognition.
        
        Args:
            user_message: The user's natural language request
            
        Returns:
            Dictionary with tool_name and arguments, or None if no tool needed
        """
        message_lower = user_message.lower().strip()
        logger.info(f"🔍 Parsing user intent from: '{user_message}'")
        
        # WEATHER INTENT DETECTION
        weather_keywords = ["weather", "temperature", "rain", "sunny", "cloudy", "forecast", "climate"]
        if any(keyword in message_lower for keyword in weather_keywords):
            logger.info("🌤️ Weather intent detected")
            
            # Extract location from message
            location = self._extract_location(user_message)
            
            # Extract units preference
            units = "fahrenheit" if any(unit in message_lower for unit in ["fahrenheit", "°f", "f"]) else "celsius"
            
            if location:
                return {
                    "tool_name": "find_weather",
                    "arguments": {"location": location, "units": units}
                }
        
        # MATH INTENT DETECTION
        math_keywords = ["calculate", "math", "compute", "solve"]
        math_operators = ["+", "-", "*", "/", "=", "^", "sqrt", "sin", "cos", "tan", "log"]
        
        if (any(keyword in message_lower for keyword in math_keywords) or 
            any(operator in user_message for operator in math_operators)):
            logger.info("🧮 Math intent detected")
            
            # Extract mathematical expression
            expression = self._extract_math_expression(user_message)
            if expression:
                return {
                    "tool_name": "calculate_math",
                    "arguments": {"expression": expression}
                }
        
        # SYSTEM INFO INTENT DETECTION
        system_keywords = ["system", "info", "computer", "memory", "cpu", "disk", "performance"]
        if any(keyword in message_lower for keyword in system_keywords):
            logger.info("💻 System info intent detected")
            return {
                "tool_name": "get_system_info",
                "arguments": {}
            }
        
        # WEB SEARCH INTENT DETECTION
        search_keywords = ["search", "find", "look up", "google", "web"]
        if any(keyword in message_lower for keyword in search_keywords):
            logger.info("🔍 Search intent detected")
            
            # Extract search query
            query = self._extract_search_query(user_message)
            if query:
                return {
                    "tool_name": "search_web",
                    "arguments": {"query": query}
                }
        
        logger.info("❓ No specific tool intent detected")
        return None
    
    def _extract_location(self, message: str) -> Optional[str]:
        """Extract location from user message using various patterns."""
        words = message.split()
        
        # Look for prepositions followed by location
        location_prepositions = ["in", "for", "at", "from"]
        for i, word in enumerate(words):
            if word.lower() in location_prepositions and i + 1 < len(words):
                # Take words after preposition until we hit another preposition or end
                location_words = []
                for j in range(i + 1, len(words)):
                    if words[j].lower() in location_prepositions:
                        break
                    location_words.append(words[j])
                if location_words:
                    return " ".join(location_words).strip(".,!?")
        
        # Look for capitalized words (likely place names)
        potential_locations = [word for word in words if word[0].isupper() and len(word) > 2 and word.isalpha()]
        if potential_locations:
            return " ".join(potential_locations)
        
        return None
    
    def _extract_math_expression(self, message: str) -> Optional[str]:
        """Extract mathematical expression from user message."""
        # Look for mathematical patterns
        math_pattern = r'[0-9+\-*/().\s]+'
        matches = re.findall(math_pattern, message)
        
        if matches:
            # Find the longest match that contains operators
            expression = max(matches, key=len).strip()
            if any(op in expression for op in ['+', '-', '*', '/', '(', ')']):
                return expression
        
        # Look for function calls like sqrt(16), sin(30)
        function_pattern = r'(sqrt|sin|cos|tan|log)\s*\(\s*[0-9.]+\s*\)'
        func_match = re.search(function_pattern, message, re.IGNORECASE)
        if func_match:
            return func_match.group(0)
        
        return None
    
    def _extract_search_query(self, message: str) -> Optional[str]:
        """Extract search query from user message."""
        message_lower = message.lower()
        
        # Remove search keywords and extract the rest
        search_patterns = [
            r'search\s+for\s+(.+)',
            r'look\s+up\s+(.+)',
            r'find\s+(.+)',
            r'google\s+(.+)',
            r'search\s+(.+)'
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1).strip(".,!?")
        
        # If no pattern matches, use the whole message minus search keywords
        words = message.split()
        filtered_words = [word for word in words if word.lower() not in ["search", "find", "look", "up", "google", "web"]]
        if filtered_words:
            return " ".join(filtered_words)
        
        return None
    
    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        This is the main entry point for the AI agent.
        
        The flow is:
        1. Parse user intent to see if tools are needed
        2. If tools are needed, call the appropriate tool via MCP server
        3. Format the tool response into human-readable text
        4. If no tools needed, provide a general conversational response
        
        Args:
            user_message: The user's message/request
            
        Returns:
            String response from the agent
        """
        logger.info(f"💬 Processing user message: '{user_message}'")
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Parse user intent to see if we need to use tools
        intent = self._parse_user_intent(user_message)
        
        if intent:
            logger.info(f"🎯 Intent detected: {intent['tool_name']}")
            
            # User wants to use a tool - call it via MCP server
            tool_result = await self.mcp_server.call_tool(
                intent["tool_name"], 
                intent["arguments"]
            )
            
            if tool_result["success"]:
                # Format the tool result into a human-readable response
                response = self._format_tool_response(intent["tool_name"], tool_result["result"])
            else:
                response = f"I encountered an error while using the {intent['tool_name']} tool: {tool_result.get('error', 'Unknown error')}"
        
        else:
            # No tool needed, provide a general conversational response
            response = self._generate_general_response(user_message)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        logger.info("✅ Message processed successfully")
        return response
    
    def _generate_general_response(self, user_message: str) -> str:
        """Generate a response when no specific tool is needed."""
        message_lower = user_message.lower()
        
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return ("Hello! 👋 I'm an AI agent that can help you with various tasks. "
                   "I can check weather, perform calculations, get system information, and search the web. "
                   "What would you like to know?")
        
        elif any(word in message_lower for word in ["help", "what can you do", "capabilities"]):
            tools = self.mcp_server.list_tools()
            tool_descriptions = [f"🔧 {tool['name']}: {tool['description']}" for tool in tools]
            return ("I can help you with the following tools:\n\n" + 
                   "\n".join(tool_descriptions) + 
                   "\n\nJust ask me naturally, like 'What's the weather in Paris?' or 'Calculate 15 * 23'")
        
        elif any(word in message_lower for word in ["thanks", "thank you", "appreciate"]):
            return "You're welcome! 😊 Is there anything else I can help you with?"
        
        elif any(word in message_lower for word in ["bye", "goodbye", "see you"]):
            return "Goodbye! 👋 Feel free to come back anytime if you need help with weather, calculations, or other tasks!"
        
        else:
            return ("I'm not sure how to help with that specific request. "
                   "I can help you with weather information, mathematical calculations, "
                   "system information, or web searches. Try asking something like:\n"
                   "• 'What's the weather in Tokyo?'\n"
                   "• 'Calculate sqrt(144) + 5'\n"
                   "• 'Show me system information'\n"
                   "• 'Search for Python tutorials'")
    
    def _format_tool_response(self, tool_name: str, result: Any) -> str:
        """
        Format tool results into human-readable responses.
        Each tool type gets its own formatting to make the output user-friendly.
        
        Args:
            tool_name: Name of the tool that was called
            result: The result returned by the tool
            
        Returns:
            Formatted string response
        """
        logger.info(f"📝 Formatting response for tool: {tool_name}")
        
        if tool_name == "find_weather":
            weather = result
            response = f"🌤️ **Weather in {weather['location']}**\n\n"
            response += f"🌡️ **Temperature:** {weather['temperature']}°{weather['units'][0].upper()}\n"
            response += f"☁️ **Condition:** {weather['condition']}\n"
            response += f"💧 **Humidity:** {weather['humidity']}%\n"
            response += f"💨 **Wind Speed:** {weather['wind_speed']} km/h\n\n"
            response += f"📅 **3-Day Forecast:**\n"
            for day in weather['forecast']:
                response += f"  • {day['day']}: {day['high']}°/{day['low']}° - {day['condition']}\n"
            return response
        
        elif tool_name == "calculate_math":
            if result["success"]:
                return f"🧮 **Calculation Result**\n\nExpression: `{result['expression']}`\nResult: **{result['result']}**"
            else:
                return f"❌ **Calculation Error**\n\nI couldn't calculate `{result['expression']}`\nError: {result['error']}"
        
        elif tool_name == "get_system_info":
            info = result
            response = f"💻 **System Information**\n\n"
            response += f"🖥️ **Platform:** {info.get('platform', 'Unknown')} {info.get('platform_release', '')}\n"
            if 'python_version' in info:
                response += f"🐍 **Python:** {info['python_version']}\n"
            if 'cpu_count' in info:
                response += f"⚙️ **CPU Cores:** {info['cpu_count']}\n"
            if 'cpu_usage_percent' in info:
                response += f"📊 **CPU Usage:** {info['cpu_usage_percent']}%\n"
            if 'memory_total_gb' in info:
                response += f"💾 **Total Memory:** {info['memory_total_gb']} GB\n"
            if 'memory_available_gb' in info:
                response += f"💾 **Available Memory:** {info['memory_available_gb']} GB\n"
            if 'memory_usage_percent' in info:
                response += f"📊 **Memory Usage:** {info['memory_usage_percent']}%\n"
            if 'disk_usage_percent' in info:
                response += f"💽 **Disk Usage:** {info['disk_usage_percent']}%\n"
            response += f"\n⏰ **Timestamp:** {info.get('timestamp', 'Unknown')}"
            return response
        
        elif tool_name == "search_web":
            search = result
            response = f"🔍 **Search Results for '{search['query']}'**\n\n"
            response += f"Found {search['results_count']} results:\n\n"
            
            for i, result_item in enumerate(search['results'], 1):
                response += f"**{i}. {result_item['title']}**\n"
                response += f"🔗 {result_item['url']}\n"
                response += f"📄 {result_item['snippet']}\n\n"
            
            return response
        
        # Fallback for unknown tool types
        return f"🔧 **Tool Result from {tool_name}:**\n\n{json.dumps(result, indent=2)}"

# ============================================================================
# STEP 5: Demo and Testing Functions
# ============================================================================

async def run_demo():
    """
    Demonstration function that shows how the AI agent system works.
    This walks through the complete flow from initialization to tool execution.
    """
    print("🤖 AI Agent with MCP Tools - Demo Starting...")
    print("=" * 60)
    
    # Step 1: Initialize the MCP Server with our custom tools
    print("📡 Step 1: Initializing MCP Server...")
    mcp_server = MCPServer()
    
    # Step 2: Create the AI Agent and connect it to the MCP server
    print("🧠 Step 2: Creating AI Agent...")
    agent = AIAgent(mcp_server)
    
    # Step 3: Show available tools
    print("\n🔧 Step 3: Available Tools:")
    tools = mcp_server.list_tools()
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool['name']}: {tool['description']}")
    
    # Step 4: Run test conversations
    print("\n" + "=" * 60)
    print("💬 Step 4: Demo Conversations")
    print("=" * 60)
    
    # Test different types of requests
    test_messages = [
        "Hello! What can you help me with?",
        "What's the weather like in New York?",
        "Can you calculate 15 * 23 + sqrt(144)?",
        "Show me system information",
        "What's the weather in Tokyo in fahrenheit?",
        "Search for Python machine learning tutorials",
        "Calculate sin(30) + cos(60)",
        "Thanks for your help!"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"👤 User: {message}")
        
        # Process the message and get response
        response = await agent.process_message(message)
        print(f"🤖 Agent: {response}")
        
        # Add a small delay to make the demo feel more natural
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("🎯 The AI agent successfully:")
    print("  • Understood natural language requests")
    print("  • Selected appropriate tools for each task")
    print("  • Executed tools via the MCP server")
    print("  • Formatted responses in a user-friendly way")

async def interactive_mode():
    """
    Run the agent in interactive mode where users can type messages.
    This provides a command-line interface to test the agent.
    """
    print("🤖 AI Agent Interactive Mode")
    print("=" * 40)
    print("Type your messages and press Enter.")
    print("Commands:")
    print("  • 'quit' or 'exit' - Exit the program")
    print("  • 'help' - Show available tools")
    print("  • 'tools' - List all registered tools")
    print("=" * 40)
    
    # Initialize system
    mcp_server = MCPServer()
    agent = AIAgent(mcp_server)
    
    print("\n🚀 System ready! How can I help you today?\n")
    
    try:
        while True:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the AI agent! Goodbye!")
                break
            
            if user_input.lower() == 'tools':
                tools = mcp_server.list_tools()
                print("\n🔧 Available Tools:")
                for i, tool in enumerate(tools, 1):
                    print(f"  {i}. {tool['name']}: {tool['description']}")
                print()
                continue
            
            if not user_input:
                continue
            
            # Process the message
            print("🤖 Agent: ", end="")
            response = await agent.process_message(user_input)
            print(response)
            print()
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

# ============================================================================
# STEP 6: Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Entry point of the program.
    This determines whether to run in demo mode or interactive mode.
    """
    import sys
    
    # Check for required dependencies
    try:
        import psutil
        print("✅ psutil available - full system information will be provided")
    except ImportError:
        print("⚠️ Warning: psutil not installed. System info tool will have limited functionality.")
        print("💡 Install with: pip install psutil")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            # Run in interactive mode
            print("Starting interactive mode...")
            asyncio.run(interactive_mode())
        elif sys.argv[1] == "demo":
            # Run demo mode
            print("Starting demo mode...")
            asyncio.run(run_demo())
        else:
            print("Usage:")
            print("  python basic_test.py demo        - Run demonstration")
            print("  python basic_test.py interactive - Run interactive mode")
            print("  python basic_test.py             - Run demonstration (default)")
    else:
        # Default to demo mode
        asyncio.run(run_demo())
