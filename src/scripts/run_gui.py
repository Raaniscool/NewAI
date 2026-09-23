#!/usr/bin/env python3
"""
ScratchLM Desktop Application GUI Launcher.

Launches the native desktop runner for ScratchLM Tiny models.
"""

import sys
from pathlib import Path

# Add src directory to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratchlm.gui import main

if __name__ == "__main__":
    main()
