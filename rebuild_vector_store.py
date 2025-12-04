#!/usr/bin/env python3
"""
Rebuild vector store with all documents in the data folder
"""

from docx_rag_pipeline import DocxExcelRagPipeline
from pathlib import Path
import json

def rebuild_vector_store():
    """Rebuild the vector store with all documents."""
    
    print("=" * 60)
    print("Rebuilding Vector Store")
    print("=" * 60)
    
    # Initialize the pipeline
    print("\n1. Initializing pipeline...")
    pipeline = DocxExcelRagPipeline(
        chroma_db_path="chroma_db",
        chunk_size=1000,
        retrieval_count=6,
        temperature=0.2,
        max_response_tokens=1200
    )
    
    # Find all .docx files
    data_folder = Path("data")
    docx_files = sorted(list(data_folder.glob("*.docx")))
    
    if not docx_files:
        print(f"\n⚠️  No .docx files found in '{data_folder}' folder.")
        return
    
    print(f"\n2. Found {len(docx_files)} document(s):")
    for i, doc in enumerate(docx_files, 1):
        print(f"   {i:2d}. {doc.name}")
    
    # Build vector store (reset=True to rebuild from scratch)
    print(f"\n3. Building vector store (this may take a few minutes)...")
    try:
        pipeline.build_vector_store(
            documents=[str(f) for f in docx_files],
            reset=True  # Rebuild from scratch
        )
        print("   ✓ Vector store built successfully!")
    except Exception as e:
        print(f"   ✗ Error building vector store: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get collection statistics
    print(f"\n4. Collection Statistics:")
    stats = pipeline.get_collection_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "=" * 60)
    print("✓ Vector store rebuild complete!")
    print("=" * 60)

if __name__ == "__main__":
    rebuild_vector_store()

