#!/usr/bin/env python3
"""
Python Build Script for Packaging ScratchLM into a Standalone Executable.
"""

import sys
import subprocess
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    spec_file = root / "ScratchLM.spec"
    dist_dir = root / "dist"
    build_dir = root / "build"

    print("==========================================================")
    print(" Building ScratchLM Standalone Executable")
    print("==========================================================")

    print("\n1. Verifying PyInstaller installation...")
    try:
        import PyInstaller
    except ImportError:
        print("Installing pyinstaller package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    print("\n2. Cleaning build directories...")
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    if dist_dir.exists():
        shutil.rmtree(dist_dir, ignore_errors=True)

    print("\n3. Executing PyInstaller compilation...")
    cmd = [sys.executable, "-m", "PyInstaller", "--clean", str(spec_file)]
    subprocess.check_call(cmd, cwd=root)

    target_exe = dist_dir / ("ScratchLM.exe" if sys.platform == "win32" else "ScratchLM")
    if target_exe.exists():
        print("\n==========================================================")
        print(" SUCCESS! Binary build complete:")
        print(f" Executable: {target_exe}")
        print(" Workflow: Double-click executable to launch ScratchLM!")
        print("==========================================================")
    else:
        print(f"\nBuild complete. Output binary located in: {dist_dir}")


if __name__ == "__main__":
    main()
