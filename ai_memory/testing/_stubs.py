import hashlib
import numpy as np

class FakeSentenceTransformer:
    """Simple fake SentenceTransformer for offline tests."""

    def __init__(self, *args, **kwargs):
        self.dim = 1024

    def _vec(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(self.dim, dtype=np.float32)

    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        vecs = np.vstack([self._vec(s) for s in sentences])
        return vecs
