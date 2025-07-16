class MemoryCompressor:
    def __init__(self):
        pass

    def _compress_code(self, code: str) -> str:
        # naive code compression: remove empty lines
        lines = [
            l
            for l in code.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        return " ".join(lines)

    def _compress_conversation(self, text: str) -> str:
        # placeholder summarization
        sentences = text.split(".")
        return ". ".join(sentences[:2])
