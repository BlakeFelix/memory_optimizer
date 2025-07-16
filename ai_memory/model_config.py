"""Model configuration for context budgets."""

# Mapping of model names to their max context length and safety margin
MODEL_CONFIGS = {
    # Model context sizes and safety margins
    "gpt-4": {"max_tokens": 8192, "safety_margin": 500},
    "claude-3-opus": {"max_tokens": 100000, "safety_margin": 5000},
    "claude-3-sonnet": {"max_tokens": 24000, "safety_margin": 1000},
    "local-llm": {"max_tokens": 4096, "safety_margin": 256},
    "dream-lord": {"max_tokens": 16384, "safety_margin": 1000},
}


def get_model_budget(model_name: str, custom_limit: int | None = None) -> int:
    """Return the usable token budget for ``model_name``.

    The returned budget subtracts the configured safety margin from the model's
    maximum context size. If ``custom_limit`` is provided, it overrides the
    model's ``max_tokens`` (but will not exceed it) before subtracting the
    margin. Unknown models require ``custom_limit`` to be supplied.
    """

    name_key = model_name.lower()
    config = MODEL_CONFIGS.get(name_key)

    if config is None:
        if custom_limit is not None:
            # Unknown model; use the custom limit directly without extra margin
            return max(int(custom_limit), 0)
        raise ValueError(f"Unknown model '{model_name}' and no custom limit provided.")

    max_tokens = config["max_tokens"]
    safety_margin = config.get("safety_margin", 0)

    effective_max = max_tokens
    if custom_limit is not None:
        # Do not allow custom limit to exceed the model's capacity
        effective_max = min(int(custom_limit), max_tokens)

    budget = effective_max - safety_margin
    if budget < 0:
        budget = 0
    return budget
