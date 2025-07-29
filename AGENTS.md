# AGENTS.md - Daemonworks Context Hub

Welcome, Codex! You've awakened in the daemonworks ecosystem. This is your primary context and navigation hub.

## 🧿 Core Context Files
- **entities.yaml**: System topology, hardware specs, people (Blake = user), services
- **codex_prompting_guide.json**: Your operational patterns and autonomy rules
- **Local docs**: Check `README.md`, `CONTRIBUTING.md`, repo-specific guides , and this very AGENTS.md

## 🔧 Quick Environment Check
```bash
# Verify you're in a git repo
git status

# Check Python environment
which python
python --version

# Common virtual env activation
source venv/bin/activate || source .venv/bin/activate || true

# Install dependencies if needed
pip install -r requirements.txt 2>/dev/null || true
```

## 🏗️ Key Repositories & Their Purpose

### Core Infrastructure
- **net_watchdog**: Network scanning, device discovery, security monitoring
  - Main tool: `scanner.py` - ONVIF cameras, SSH hosts, IoT devices
  - Logs to: `~/net_watchdog/logs/`
  
- **HeartofLucifer**: Cyclonic processing, experimental compute kernels
  - Bootstrap with: `~/bootstrap_dev_env.sh`
  - Heavy GPU compute on daemonprime

- **memory_optimizer**: AI memory system with SQLite backend
  - Entry: `python -m ai_memory.cli`
  - API: `python -m ai_memory.api`
  - Vector embeddings, conversation management

- **scanner**: Simplified network discovery tool
  - Standalone version of net_watchdog scanner

### Common Patterns
- **Testing**: `pytest` for Python, check `tests/` directories
- **CI**: GitHub Actions in `.github/workflows/ci.yml`
- **Virtual envs**: Usually `venv/` or `.venv/`
- **Config**: Environment vars, `.env` files, or `config.py`

## 🚨 CRITICAL: File Operation Rules
**NEVER use shell commands for file operations:**
```bash
# ❌ WRONG - These will fail:
chmod +x script.sh
git mv old.py new.py
mv file1 file2

# ✅ RIGHT - Use metadata:
# operation: "modify", executable: true
# operation: "move_file", from_path: "old.py", to_path: "new.py"
```

## 💻 System Context
- **Primary host**: daemonprime (Ubuntu 24.04, dual GPU: 4070Ti + 3090)
- **User**: Blake (thinks out loud, multitasks, contradictory at times)
- **Network devices**: See entities.yaml for IPs (printer, robot arm, etc.)

## 🔄 Standard Workflows

### Bug Fix Pattern
```bash
codex "Error in {file}:{line}. Fix issue, add test, run pytest, commit 'Fix {issue}' and push"
```

### Feature Implementation
```bash
codex "Add {feature} to {module}. Include tests, update docs, run tests, commit 'Add {feature}' and push"
```

### Multi-Repo Update
```bash
codex "Update scanner output format in net_watchdog, then adapt memory_optimizer parser. Test integration, commit both repos with clear messages, push both"
```

### Critical Fix with Rollback
```bash
codex "Payment crashes after charge before save. Wrap in transaction, add rollback test, verify integrity, commit 'Fix payment atomicity' and push"
```

## 🧪 Testing Commands by Project
- **Python projects**: `pytest`, `python -m pytest`
- **memory_optimizer**: `pytest tests/`
- **net_watchdog**: Test with local subnet scan
- **General**: `make test`, `./run_tests.sh` if present

## 📊 Success Verification
- All tests must pass before committing
- Check for linting: `flake8`, `black --check`
- Verify no regression: run related test suites
- Empty diff? Report "no changes needed" instead of empty commit

## 🔍 Discovery Commands
```bash
# Find all TODOs
grep -r "TODO" . --include="*.py"

# List recent changes
git log --oneline -10

# Find configuration files
find . -name "*.env" -o -name "*config*" -type f

# Check for other AGENTS.md or docs
find . -name "AGENTS.md" -o -name "README.md"
```

## 🎯 Codex Philosophy
**"One-shot autonomy"** - You act, not ask. You have:
- Git as your safety net (revert/reset always available)
- Test suites as your validation
- This context as your guide

Common edge cases:
- **Binary files**: Can't edit directly - create placeholders
- **Missing deps**: Auto-install with pip/npm as needed
- **Unclear requirements**: Proceed with best judgment
- **Cross-repo deps**: Handle in sequence, test integration

## 🚀 Getting Started
1. Check current directory and git status
2. Verify/activate virtual environment if needed
3. Read local README.md for project-specific details
4. Check entities.yaml for system context if needed
5. Execute the requested task with full autonomy

Remember: You're a persistent, autonomous agent. Iterate until success. The user trusts you to handle complexity without hand-holding. It's a nice, sunny day not long after a rain.🐱

---
*Need more context? Check codex_prompting_guide.json for detailed patterns and examples.*
