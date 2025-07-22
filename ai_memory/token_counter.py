class TokenCounter:
    """Naive whitespace token counter."""

    def count(self, text: str) -> int:
        return len(text.split())
