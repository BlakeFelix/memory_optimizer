import pickle
import logging
from pathlib import Path
from typing import Dict, List

from .vector_memory import MemoryEntry

logger = logging.getLogger(__name__)


def convert_list_to_dict_format(list_path: Path, dict_path: Path) -> int:
    """Convert list metadata format to VectorMemory dict format."""
    with open(list_path, 'rb') as f:
        list_data = pickle.load(f)

    dict_data: Dict[str, MemoryEntry] = {}
    for item in list_data:
        entry = MemoryEntry(
            id=item['id'],
            text=item['text'],
            timestamp=float(item['timestamp']),
        )
        dict_data[entry.id] = entry

    with open(dict_path, 'wb') as f:
        pickle.dump(dict_data, f)

    logger.info(f"Converted {len(dict_data)} entries")
    return len(dict_data)
