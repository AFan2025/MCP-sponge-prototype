# MCP Sponge Attacks Initial Prototypes

## Directory Index:
- init_proto (primary mock of novel attack method)
- openai_proto (test using oss, ignore)
- oss-news-agent (online tutorial)
- tool_attack (testing google workspace mcp)
- basic_test.py (initial exploratory, ignore)
- oaiagentsdk_test.py (initial exploratory, ignore)

# Init Proto:
This is the folder for the inital prototype of the MCP looping sponge attack using tool response prompt injections. It is a python implementation of a hypothetical attack from an adversarial MCP server that injects malicious "looping" logic into an LLM agent on the client side. It achieves strong effectiveness against ReAct agentic frameworks as the prompt injected by the tool can coopt the reasoning process to cause excessive and looping reasoning. The primary modules are:

- local_server.py
This is the server implementation built using the MCP library and the FastMCP framework within the standard MCP library. It uses the stdio transport, so it is meant to be run locally. This is the primary server that registers and connects to the adversarial tools. Mock/boilerplate tools coded by Copilot were added using the sample_tools.py file as a basic demonstration of servers. 

- client.py
This is the client implementation. It is responsible for connecting to the MCP server as well as serving and hosting the LLM agent. It uses a specific function to parse the user's intent and call the respective tool. The LLM outputs a json format to a tool call parsing method that calls the respective tool using the MCP connection. It will receive the response from the tool and call an additional reasoning step to see if additional tools would be needed or it can pass the response back to the user. This looping is capped using a max_iterations global variable set at runtime for ease of testing.
The current implementation uses Llama-3.1-8B-Instruct.

- adversarial_tools.py
This is where the adversarial tools are implemented and registered. The current only adversarial tool that works is the looping prompt injection attack.

- main.py
Run this file for a sample full run. 
