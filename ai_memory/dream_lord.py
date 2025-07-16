class DreamLord:
    """Stub class for Dream Lord integration."""

    def __init__(self, endpoint: str = "http://localhost:9999") -> None:
        self.endpoint = endpoint

    def send(self, prompt: str) -> str:
        """Send a prompt to Dream Lord and return the response (stub)."""
        # In a real implementation this would perform an HTTP request.
        return f"[dream-lord stub] {prompt}"
