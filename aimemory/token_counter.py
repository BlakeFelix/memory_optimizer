from functools import lru_cache


class TokenCounter:
    """Token counter using tiktoken when available."""

    def __init__(self, model: str = "cl100k_base"):
        try:
            import tiktoken

            self.encoder = tiktoken.get_encoding(model)
        except Exception:  # pragma: no cover - tiktoken optional
            self.encoder = None

    @lru_cache(maxsize=2048)
    def count(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text.split())

    def estimate(self, text: str) -> int:
        """Rough estimate without encoding."""
        return max(1, len(text) // 4)
