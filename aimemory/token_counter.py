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

    def estimate(self, text: str, model: str = "default") -> int:
        """Heuristic token estimation without a tokenizer.

        The estimation is based on character count with adjustments for
        punctuation, code blocks and URLs. It loosely assumes that four
        characters correspond to roughly one token.
        """
        import re

        char_count = len(text)

        # URLs tend to use more tokens due to slashes and parameters
        if re.search(r"https?://", text):
            char_count *= 1.1

        # Code blocks expand slightly because of formatting characters
        if "```" in text:
            char_count *= 1.2

        # punctuation adds token boundaries
        punctuation = len(re.findall(r"[.!?]", text))
        char_count += punctuation

        # naive multilingual adjustment for non-ascii text
        if re.search(r"[^\x00-\x7F]", text):
            char_count *= 1.1

        multipliers = {"gpt": 1.0, "claude": 1.1, "gemini": 1.2}
        model_mul = multipliers.get(model.lower(), 1.0)

        return max(1, int(char_count / 4 * model_mul))
