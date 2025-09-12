#!/usr/bin/env python3
"""
Test script to validate the integration fixes in client.py
"""

import sys
import os
import asyncio
import logging
from unittest.mock import Mock, MagicMock, patch

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the MCP imports since they're not installed yet
sys.modules['mcp'] = Mock()
sys.modules['mcp.server'] = Mock()
sys.modules['mcp.transport'] = Mock()
sys.modules['mcp.transport.StdioTransport'] = Mock()

def test_basic_imports():
    """Test that the main imports work"""
    try:
        # Import the basic modules that should work
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_device_handling():
    """Test device detection logic"""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Device detection successful: {device}")
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_client_structure():
    """Test that client.py has the right structure"""
    try:
        # Mock MCP components more thoroughly
        with patch.dict('sys.modules', {
            'mcp': Mock(),
            'mcp.server': Mock(),
            'mcp.transport': Mock(),
            'mcp.types': Mock(),
        }):
            from client import MCPAIAgent
            print("✅ Client class import successful")
            return True
    except Exception as e:
        print(f"❌ Client import failed: {e}")
        return False

async def test_async_methods():
    """Test that async methods are properly defined"""
    try:
        with patch.dict('sys.modules', {
            'mcp': Mock(),
            'mcp.server': Mock(), 
            'mcp.transport': Mock(),
            'mcp.types': Mock(),
        }):
            from client import MCPAIAgent
            
            # Create a mock instance
            agent = MCPAIAgent(use_local_llm=False)  # Don't load model for testing
            
            # Check if parse_user_intent is async
            import inspect
            is_async = inspect.iscoroutinefunction(agent.parse_user_intent)
            if is_async:
                print("✅ parse_user_intent is correctly async")
                return True
            else:
                print("❌ parse_user_intent is not async")
                return False
                
    except Exception as e:
        print(f"❌ Async method test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing integration fixes...\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Device Handling", test_device_handling),
        ("Client Structure", test_client_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    # Run async test separately
    print("Running Async Methods test...")
    async_result = asyncio.run(test_async_methods())
    results.append(async_result)
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All integration fixes validated!")
    else:
        print("⚠️  Some issues remain")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
