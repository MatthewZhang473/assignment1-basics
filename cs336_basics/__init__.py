import importlib.metadata
from .pretokenization_example import find_chunk_boundaries

__version__ = importlib.metadata.version("cs336_basics")
__all__ =['find_chunk_boundaries']