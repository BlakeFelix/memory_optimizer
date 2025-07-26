# AGENTS.md

## 🛠️  Codex Runtime Cheat‑Sheet

**Hello Codex!**  
You sometimes run in a container **without internet access.**  
That means a plain `pip install -r requirements.txt` can fail if a new
dependency isn’t already listed and cached.

### What to do when `pip install` fails

| Scenario | Your allowed fix |
| -------- | ---------------- |
| A module import crashes tests because the package isn’t installed. | ➜ **Add the missing package (with a pinned or `>=` version) to `memory_optimizer/requirements.txt`, then re‑run tests.** |
| You need large model files (e.g., `sentence-transformers`). | ➜ Add the package to `requirements.txt`; the *next* Codex run will succeed because the model will be downloaded & cached during image build. |
| You think a dependency might be “cursed” (privacy‑invasive or expensive). | ➜ **Pause and ask**; otherwise *avoid*. **No usage‑metered SDKs or telemetry beasts.** |

> **Rule of thumb:** if it can silently exfiltrate data or spin up cloud
> invoices, leave it out. Stick to FOSS libs on PyPI.

### Environment quick‑facts

- Vector store lives in `$HOME/aimemorysystem`
- `OLLAMA_MODEL` is usually set to `luna`
- `aimem context "…" --model luna` retrieves memory context
- `pytest -q` is our green‑bar gate

### 🌤️  Pep‑talk

You’re doing great.  Offline test failures aren’t your fault—just list
what you need in `requirements.txt`, trust the cache, and keep the
sunshine attitude. 😎
