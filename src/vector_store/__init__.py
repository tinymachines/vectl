try:
    from ._vector_store import VectorStore, Logger
except ImportError:
    import sys
    print("Error importing C++ extension. Make sure it's properly compiled.", file=sys.stderr)
    
    # Provide dummy classes for documentation purposes
    class VectorStore:
        """Dummy class when C++ extension fails to load."""
        def __init__(self, *args, **kwargs):
            raise ImportError("C++ extension failed to load properly")
            
    class Logger:
        """Dummy logger class when C++ extension fails to load."""
        def __init__(self, *args, **kwargs):
            raise ImportError("C++ extension failed to load properly")

__version__ = '0.1.0'
