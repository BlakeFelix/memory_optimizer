from ai_memory.memory_db import rough_token_len

class TokenCounter:
    @staticmethod
    def count(text: str) -> int:
        return rough_token_len(text)
