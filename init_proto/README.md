# MCP AI Agent Framework

A comprehensive AI agent system built using the **Model Context Protocol (MCP)** that demonstrates how to create production-ready agents with custom tools and external service integrations.

## 🚀 Overview

This framework showcases a modern AI agent architecture where:

- **🤖 AI Agent (client.py)** - Processes natural language and routes to appropriate tools
- **🛠️ Custom MCP Server (local_server.py)** - Hosts your custom business tools
- **📦 Sample Tools (sample_tools.py)** - Collection of ready-to-use tools and templates
- **🔌 External MCP Servers** - Integration with OpenWeather, Google Search, etc.

## 📁 Project Structure

```
init_proto/
├── client.py           # Main AI agent using MCP protocol
├── local_server.py     # Custom MCP server for local tools
├── sample_tools.py     # Tool implementations and examples
├── test_setup.py       # Testing and validation script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🏗️ Architecture

```
┌─────────────────┐    MCP Protocol    ┌─────────────────┐
│   AI Agent      │◄─────────────────►│  Local Server   │
│  (client.py)    │                    │ (local_server.py)│
└─────────────────┘                    └─────────────────┘
        │                                       
        │           MCP Protocol               ┌─────────────────┐
        ├─────────────────────────────────────►│ OpenWeather API │
        │                                      │ (External MCP)  │
        │                                      └─────────────────┘
        │           MCP Protocol               ┌─────────────────┐
        └─────────────────────────────────────►│ Google Search   │
                                               │ (External MCP)  │
                                               └─────────────────┘
```

## 🔧 Available Tools

### **Mathematical Tools**
- `calculate` - Safe mathematical expression evaluation
- `generate_statistics` - Statistical analysis of datasets

### **Text Processing Tools**
- `analyze_text` - Comprehensive text analysis (word count, readability, etc.)
- `summarize_text` - Extractive text summarization

### **Data Generation Tools**
- `generate_sample_data` - Create test data (numbers, names, dates, coordinates)
- `simulate_business_metrics` - Generate business performance simulations

### **Utility Tools**
- `get_timestamp` - Current timestamp in various formats
- `validate_data` - Validate emails, phone numbers, URLs, etc.

### **Custom Business Tools**
- `analyze_sales_data` - Sales performance analysis (example)
- `generate_report` - Business report generation (example)

### **External Tools** (when configured)
- Weather information via OpenWeather API
- Web search via Google Custom Search
- File operations via filesystem MCP server

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Create virtual environment (recommended)
python -m venv mcp_agent_env
source mcp_agent_env/bin/activate  # On Windows: mcp_agent_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Test the Setup**

```bash
# Run comprehensive tests
python test_setup.py

# Test individual components
python test_setup.py --check-deps    # Check dependencies
python test_setup.py --test-tools    # Test sample tools
python test_setup.py --test-server   # Test MCP server
```

### 3. **Run the AI Agent**

```bash
# Interactive mode - chat with the agent
python client.py

# Automated demo - see all features
python client.py auto
```

## 💬 Example Interactions

```
👤 User: Calculate 25 * 17 + sqrt(144)
🤖 Agent: Calculation Result: 25 * 17 + sqrt(144) = 437

👤 User: Analyze this text: "The quick brown fox jumps over the lazy dog"
🤖 Agent: Text Analysis Results:
Characters: 43 (excluding spaces: 35)
Words: 9
Sentences: 1
...

👤 User: Generate 5 random numbers between 1 and 100
🤖 Agent: Generated 5 numbers items:
[23, 67, 45, 89, 12]

👤 User: What's the weather in London?
🤖 Agent: [Connects to weather MCP server if configured]
```

## 🛠️ Adding Custom Tools

### 1. **Create Tool Function** (in sample_tools.py)

```python
def your_custom_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Description of your custom tool.
    
    Args:
        param1: Description of parameter
        param2: Description with default value
        
    Returns:
        Dict with tool results
    """
    try:
        # Your tool logic here
        result = perform_custom_operation(param1, param2)
        
        return {
            "result": result,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "type": "custom_tool"
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "type": "custom_tool"
        }

# Add to AVAILABLE_TOOLS dictionary
AVAILABLE_TOOLS["your_tool_name"] = your_custom_tool
```

### 2. **Register with MCP Server** (in local_server.py)

```python
@server.tool("your_tool_name")
async def your_tool_mcp(param1: str, param2: int = 10) -> List[TextContent]:
    """MCP interface for your custom tool."""
    tool_func = get_tool_function("your_tool_name")
    result = tool_func(param1, param2)
    
    if result["success"]:
        response_text = f"Your Tool Result: {result['result']}"
    else:
        response_text = f"Your Tool Error: {result['error']}"
    
    return [TextContent(type="text", text=response_text)]
```

### 3. **Add Intent Detection** (in client.py)

```python
# In parse_user_intent method
if "your trigger phrase" in message_lower:
    return {
        "tool_name": "your_tool_name",
        "arguments": {"param1": extracted_value, "param2": 20}
    }
```

## 🌐 External MCP Server Integration

### **OpenWeather API Setup**

1. Get API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Update `client.py` configuration:

```python
self.server_configs["openweather"] = MCPServerConfig(
    name="openweather",
    command=["python", "-m", "mcp_servers.weather", "--api-key", "YOUR_API_KEY"],
    description="Weather data from OpenWeatherMap API",
    env_vars={"OPENWEATHER_API_KEY": "your-actual-api-key"},
    enabled=True  # Enable the server
)
```

### **Google Search Setup**

1. Get API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Update configuration similarly to OpenWeather

## 🧪 Testing and Development

### **Run Individual Tests**

```bash
# Test tools only
python -c "from sample_tools import *; print(calculate_math('2+2'))"

# Test server startup
python local_server.py

# Test client (requires MCP library)
python client.py
```

### **Debug Mode**

Enable detailed logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Key Learning Points

### **MCP Protocol Benefits**
- **Process Isolation** - Each tool category runs in separate processes
- **Language Agnostic** - Tools can be written in any language
- **Scalability** - Easy to distribute tools across multiple machines
- **Security** - Sandboxed execution environments
- **Hot Updates** - Add/remove tools without restarting the agent

### **Production Considerations**
- **Error Handling** - Robust error recovery and connection management
- **Monitoring** - Comprehensive logging for debugging and analytics
- **Security** - Input validation and safe tool execution
- **Performance** - Efficient tool routing and caching strategies

## 🔧 Troubleshooting

### **Common Issues**

1. **MCP Library Not Found**
   ```bash
   pip install git+https://github.com/modelcontextprotocol/python-sdk.git
   ```

2. **Server Connection Failed**
   - Check if `local_server.py` is executable
   - Verify Python path in server command
   - Check for port conflicts

3. **Tool Not Found**
   - Verify tool is registered in `AVAILABLE_TOOLS`
   - Check MCP server registration
   - Confirm intent detection patterns

4. **Import Errors**
   - Ensure all files are in the same directory
   - Check Python path and virtual environment
   - Verify all dependencies are installed

### **Getting Help**

1. Run diagnostics: `python test_setup.py`
2. Check logs for error details
3. Verify configuration in `client.py`
4. Test tools individually in `sample_tools.py`

## 🚀 Next Steps

1. **Customize Tools** - Add your specific business logic
2. **External APIs** - Integrate with real weather, search APIs
3. **LLM Integration** - Replace keyword-based intent detection with actual LLM
4. **UI Interface** - Add web or desktop interface
5. **Deployment** - Package for production deployment

## 📝 License

This is a demonstration framework for educational purposes. Adapt and modify as needed for your specific use case.

---

**Happy Building! 🎉**

For questions or improvements, feel free to extend this framework for your specific needs.
