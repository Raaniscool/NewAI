#!/usr/bin/env python3
"""
Setup script for ScratchLM.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scratchlm",
    version="0.1.0",
    author="Raaniscool",
    author_email="",
    description="Build a language model from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Raaniscool/NewAI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
            "types-pyyaml",
        ],
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "scratchlm-train=scripts.train:main",
            "scratchlm-evaluate=scripts.evaluate:main",
            "scratchlm-generate=scripts.generate:main",
        ],
    },
)
