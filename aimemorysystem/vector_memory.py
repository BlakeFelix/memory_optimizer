import os
from pathlib import Path
import pickle
import logging

BASE_DIR = Path(os.getenv("LUNA_VECTOR_DIR", str(Path.home() / "aimemorysystem")))
DEFAULT_INDEX = BASE_DIR / "memory_store.index"
DEFAULT_META = BASE_DIR / "memory_store.pkl"
INDEX_PATH = Path(os.getenv("LUNA_VECTOR_INDEX", DEFAULT_INDEX))
META_PATH = INDEX_PATH.with_suffix(".pkl")

BASE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def load_metadata(path: Path = META_PATH):
    if not path.exists():
        logger.warning("Metadata file %s not found", path)
        return []
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to read metadata: %s", e)
        return []

