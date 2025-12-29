#!/usr/bin/env python3
"""
Root entry point for DeskGUI.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from src.main import main

if __name__ == "__main__":
    main()
