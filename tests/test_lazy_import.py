import subprocess
import sys

def test_init_without_faiss():
    code = "\n".join([
        "import builtins, sys",
        "orig = builtins.__import__",
        "def fake(name, *a, **k):",
        "    if name == 'faiss': raise ImportError('no faiss')",
        "    return orig(name, *a, **k)",
        "builtins.__import__ = fake",
        "import ai_memory, sys",
        "print(hasattr(ai_memory, 'VectorMemory'))",
    ])
    res = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == 'True'
