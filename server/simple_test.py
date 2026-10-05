"""手动环境自检脚本:验证解释器与核心依赖可用(非自动化测试)。"""

import os
import sys

print("Hello, World!")
print("Testing Python execution...")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

try:
    import fastapi

    print(f"FastAPI version: {fastapi.__version__}")
except ImportError as e:
    print(f"FastAPI import error: {e}")

print("Simple test completed.")
