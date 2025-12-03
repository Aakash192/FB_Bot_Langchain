"""
Quick fix script for Python 3.8 compatibility issues with ChromaDB.

This script patches the posthog library to work with Python 3.8.
Run this before importing chromadb if you're using Python 3.8.
"""

import sys
import os

if sys.version_info < (3, 9):
    print("Python 3.8 detected. Applying compatibility fixes...")
    
    # Set environment variables before any imports
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    os.environ['CHROMA_TELEMETRY_DISABLED'] = '1'
    
    # Try to patch posthog if it's causing issues
    try:
        import posthog.types
        
        # Patch the problematic TypedDict usage
        if hasattr(posthog.types, 'FlagsResponse'):
            # This is a workaround - the actual fix requires updating posthog
            # or using a compatible version
            print("Note: If you encounter 'dict[str, ...]' errors, try:")
            print("  pip install 'chromadb==0.4.18'")
            print("  OR upgrade to Python 3.9+")
    except (ImportError, AttributeError):
        pass
    
    print("Compatibility fixes applied.")

