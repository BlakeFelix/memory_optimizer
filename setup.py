from setuptools import setup, find_packages

setup(
    name="ai-memory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "flask>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aimem=ai_memory.cli:cli",
        ],
    },
    python_requires=">=3.8",
    author="BlakeFelix",
    description="SQLite-backed memory optimization for LLMs",
)
