from setuptools import setup, find_packages

setup(
    name="ai-memory",
    version="0.1.5",
    packages=find_packages(include=["ai_memory", "ai_memory.*"]),
    install_requires=[
        "click>=8.0.0",
        "flask>=2.0.0",
        "faiss-cpu>=1.8.0",
    ],
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "aimem=ai_memory.cli:cli",
            "luna=ai_memory.luna_wrapper:main",
        ],
    },
    python_requires=">=3.8",
    author="BlakeFelix",
    description="SQLite-backed memory optimization for LLMs",
)
