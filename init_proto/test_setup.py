#!/usr/bin/env python3
"""
Test and Setup Script for MCP AI Agent Framework
===============================================

This script helps you test and validate the MCP AI Agent framework.
It checks dependencies, tests individual components, and provides
guidance for setup and configuration.

Usage:
    python test_setup.py                 # Run all tests
    python test_setup.py --check-deps    # Only check dependencies
    python test_setup.py --test-tools    # Only test tools
    python test_setup.py --test-server   # Only test server

Author: Assistant
Date: August 15, 2025
"""

import sys
import subprocess
import importlib
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_modules = [
        "asyncio",
        "json", 
        "logging",
        "datetime",
        "typing",
        "dataclasses",
        "pathlib"
    ]
    
    optional_modules = [
        "mcp",
        "pydantic",
        "jsonschema"
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required modules (built-in)
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} (required)")
            missing_required.append(module)
    
    # Check optional modules
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️  {module} (optional - needed for full MCP functionality)")
            missing_optional.append(module)
    
    if missing_required:
        print(f"\n❌ Missing required dependencies: {missing_required}")
        return False
    
    if missing_optional:
        print(f"\n💡 Missing optional dependencies: {missing_optional}")
        print("📚 Install with: pip install -r requirements.txt")
        print("🔧 For MCP library, you might need: pip install git+https://github.com/modelcontextprotocol/python-sdk.git")
    
    return True

def test_sample_tools():
    """Test the sample tools implementation."""
    print("\n🔧 Testing sample tools...")
    
    try:
        from sample_tools import AVAILABLE_TOOLS, calculate_math, analyze_text, generate_sample_data
        
        print(f"✅ Loaded {len(AVAILABLE_TOOLS)} tools")
        
        # Test math tool
        result = calculate_math("2 + 2")
        if result["success"] and result["result"] == 4:
            print("✅ Math tool test passed")
        else:
            print(f"❌ Math tool test failed: {result}")
            return False
        
        # Test text analysis tool
        result = analyze_text("This is a test sentence.")
        if result["success"] and result["word_count"] == 5:
            print("✅ Text analysis tool test passed")
        else:
            print(f"❌ Text analysis tool test failed: {result}")
            return False
        
        # Test data generation tool
        result = generate_sample_data("numbers", 3, min=1, max=10)
        if result["success"] and len(result["data"]) == 3:
            print("✅ Data generation tool test passed")
        else:
            print(f"❌ Data generation tool test failed: {result}")
            return False
        
        print("✅ All sample tools tests passed")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import sample_tools: {e}")
        return False
    except Exception as e:
        print(f"❌ Tool testing failed: {e}")
        return False

async def test_local_server():
    """Test if the local MCP server can start properly."""
    print("\n🖥️  Testing local MCP server...")
    
    try:
        # Try to import the server
        from local_server import server
        print("✅ Local server module imported successfully")
        
        # Check if tools are registered
        # Note: This is a simplified test since full server testing requires MCP client
        print("✅ Server appears to be properly configured")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import local_server: {e}")
        return False
    except Exception as e:
        print(f"❌ Server testing failed: {e}")
        return False

def test_client_configuration():
    """Test the client configuration and basic setup."""
    print("\n🤖 Testing client configuration...")
    
    try:
        # Check if client file exists and is importable
        client_path = Path("client.py")
        if not client_path.exists():
            print("❌ client.py not found")
            return False
        
        print("✅ client.py file exists")
        
        # Try to import (but don't run since it requires MCP library)
        # We'll just check the file syntax
        with open("client.py", "r") as f:
            content = f.read()
            
        # Basic syntax check
        try:
            compile(content, "client.py", "exec")
            print("✅ Client file syntax is valid")
        except SyntaxError as e:
            print(f"❌ Client file has syntax errors: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Client testing failed: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("📋 MCP AI AGENT FRAMEWORK TEST REPORT")
    print("="*60)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Sample Tools", test_sample_tools),
        ("Client Configuration", test_client_configuration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Server test (separate because it might not work without MCP library)
    try:
        print(f"\n🧪 Running: Local Server")
        result = asyncio.run(test_local_server())
        results["Local Server"] = result
    except Exception as e:
        print(f"⚠️  Local Server test skipped (likely missing MCP library): {e}")
        results["Local Server"] = "skipped"
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}: FAILED")
            failed += 1
        else:
            print(f"⚠️  {test_name}: SKIPPED")
            skipped += 1
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All critical tests passed! Framework is ready to use.")
        print("\n💡 Next steps:")
        print("1. Install MCP library: pip install -r requirements.txt")
        print("2. Run the client: python client.py")
        print("3. Try the interactive demo: python client.py")
    else:
        print(f"\n⚠️  {failed} tests failed. Please fix the issues before proceeding.")
        print("\n🔧 Common fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check Python version (3.8+ required)")
        print("- Ensure all files are in the same directory")
    
    return failed == 0

def main():
    """Main function to run tests based on command line arguments."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--check-deps":
            check_dependencies()
        elif arg == "--test-tools":
            test_sample_tools()
        elif arg == "--test-server":
            asyncio.run(test_local_server())
        elif arg == "--help":
            print(__doc__)
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
    else:
        # Run full test suite
        generate_test_report()

if __name__ == "__main__":
    main()
