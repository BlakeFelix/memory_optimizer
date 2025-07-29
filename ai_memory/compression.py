from __future__ import annotations

import re

from .token_counter import TokenCounter


class MemoryCompressor:
    """Stateless helpers to shrink memory fragments."""

    def __init__(self) -> None:
        self.counter = TokenCounter()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _trim(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) <= 120:
            return " ".join(tokens)
        return " ".join(tokens[:120])

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def compress_code(self, code: str) -> str:
        """Return code with comments/blank lines removed and whitespace normalised."""
        lines: list[str] = []
        for line in code.splitlines():
            # strip inline comments
            line = re.sub(r"#.*", "", line).strip()
            if not line:
                continue
            lines.append(line)
        result = " ".join(lines)
        return self._trim(result)

    def compress_text(self, text: str) -> str:
        """Summarise text using a simple frequency based heuristic."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return ""
        # build word frequencies
        freq: dict[str, int] = {}
        for sent in sentences:
            for tok in re.findall(r"\b\w+\b", sent.lower()):
                freq[tok] = freq.get(tok, 0) + 1

        scored: list[tuple[int, str]] = []
        for sent in sentences:
            score = sum(freq.get(tok, 0) for tok in re.findall(r"\b\w+\b", sent.lower()))
            scored.append((score, sent))

        top_sentences = [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]
        result = " ".join(top_sentences)
        return self._trim(result)

