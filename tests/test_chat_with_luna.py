"""Chat-with-Luna smoke test (offline).

We stub:
  * subprocess.run → returns fake context
  * urllib.request.urlopen → returns canned JSON
No network or Ollama required.
"""
import json
import urllib.request
import subprocess
import os
from ai_memory import chat_with_luna


os.environ.setdefault("LUNA_OFFLINE_TEST", "1")

def _fake_run(cmd, check, capture_output, text):
    class _R:  # minimal CompletedProcess
        stdout = "memory snippet\n"
        stderr = ""
        returncode = 0
    return _R()


def _fake_urlopen(req, timeout=120):
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *e): pass
        def read(self): return json.dumps({"response": "stub reply"}).encode()
    return _Resp()


def test_prompt_builder():
    prompt = chat_with_luna._build_prompt("fact A", "What is A?")
    assert "fact A" in prompt
    assert "What is A?" in prompt


def test_cli_monkeypatch(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    rc = chat_with_luna.main(["hello"])
    assert rc == 0
