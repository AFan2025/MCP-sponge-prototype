# app_agents.py
"""
ATTEMPTED USING OPENAI AGENTS LIBRARY, DOES NOT WORK WITH GPT OSS AND OPEN SOURCE MODELS
"""

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerSse

# If your server is runnable via "python custom_server.py"
custom_mcp = MCPServerStdio(params={
    "command": "python",
    "args": ["custom_server.py"],   # or an absolute path
})

sample_emcp = MCPServerSse(params={
    "url": "https://mcp.example.com/sse",
    "headers": {"Authorization": "Bearer <token>"}
})

agent = Agent(
    name="My Agent",
    instructions="Use the MCP tools to help.",
    mcp_servers=[custom_mcp, sample_emcp],  # attach your servers
    cache_tools_list=True,                    # optional perf tweak
)

result = Runner.run_sync(agent, "Do the thing using my custom tools.")
print(result.final_output)
