from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "memory_config.yaml"


def _parse_token_budgets(path: Path) -> dict[str, int]:
    budgets: dict[str, int] = {}
    in_block = False
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if not line.strip():
                continue
            if line.lstrip().startswith('#'):
                continue
            if line.strip() == "token_budgets:":
                in_block = True
                continue
            if in_block:
                if line.startswith("    "):
                    item = line.strip()
                    if ':' in item:
                        k, v = item.split(':', 1)
                        budgets[k.strip()] = int(v.strip())
                else:
                    break
    return budgets


def load_config(path: Path | str | None = None) -> dict:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    return {"optimization": {"token_budgets": _parse_token_budgets(cfg_path)}}


def get_model_budget(model: str, override: int | None = None) -> int:
    conf = load_config()
    budgets = conf.get("optimization", {}).get("token_budgets", {})
    if model not in budgets:
        raise KeyError(f"Unknown model: {model}")
    if override is not None:
        return int(override)
    return int(budgets[model])
