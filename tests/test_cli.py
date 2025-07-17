import subprocess, json, os, sys, tempfile, time


def _run(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_add_and_list():
    _run("python -m ai_memory.cli add 'pytest memory' -i 4.2 -c test_session")
    out = _run("python -m ai_memory.cli list -n 1 -c test_session")
    assert 'pytest memory' in out


def test_context():
    ctx = _run("python -m ai_memory.cli context 'pytest memory' --conversation-id test_session")
    assert 'pytest memory' in ctx

def test_export(tmp_path):
    out = tmp_path / "mem.json"
    _run(f"python -m ai_memory.cli export -o {out}")
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert any('pytest memory' in m['content'] for m in data)


def test_list_by_entity():
    _run("python -m ai_memory.cli add 'contact me at foo@example.com' -c entity_sess")
    out = _run("python -m ai_memory.cli list --entity foo@example.com -n 1")
    assert 'foo@example.com' in out
