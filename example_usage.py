"""
Example usage script for DOCX RAG Pipeline

This script demonstrates how to use the RAG pipeline to analyze documents.
"""

from docx_rag_pipeline import DocxExcelRagPipeline
from pathlib import Path
import json

def main():
    """Example usage of the RAG pipeline."""
    
    print("=" * 60)
    print("DOCX RAG Pipeline - Example Usage")
    print("=" * 60)
    
    # Initialize the pipeline
    print("\n1. Initializing pipeline...")
    pipeline = DocxExcelRagPipeline(
        chroma_db_path="chroma_db",
        chunk_size=1000,
        retrieval_count=3,
        temperature=0.5
    )
    
    # Check for documents in data folder
    data_folder = Path("data")
    if not data_folder.exists():
        print(f"\n⚠️  '{data_folder}' folder not found. Creating it...")
        data_folder.mkdir(exist_ok=True)
        print(f"   Please add .docx files to '{data_folder}' and run again.")
        return
    
    # Find .docx files
    docx_files = list(data_folder.glob("*.docx"))
    
    if not docx_files:
        print(f"\n⚠️  No .docx files found in '{data_folder}' folder.")
        print(f"   Please add your Word documents to '{data_folder}' and run again.")
        return
    
    print(f"\n2. Found {len(docx_files)} document(s):")
    for doc in docx_files:
        print(f"   - {doc.name}")
    
    # Build vector store
    print(f"\n3. Building vector store...")
    try:
        pipeline.build_vector_store(
            documents=[str(f) for f in docx_files],
            reset=False  # Set to True to rebuild from scratch
        )
        print("   ✓ Vector store built successfully!")
    except Exception as e:
        print(f"   ✗ Error building vector store: {e}")
        return
    
    # Get collection statistics
    print(f"\n4. Collection Statistics:")
    stats = pipeline.get_collection_stats()
    print(json.dumps(stats, indent=2))
    
    # Example queries
    print("\n" + "=" * 60)
    print("5. Example Queries")
    print("=" * 60)
    
    example_queries = [
        "What are the key financial metrics in the documents?",
        "Summarize the main findings from the documents",
        "What tables are available in the documents?",
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Question: {query}")
        print("\nAnswer:")
        
        try:
            result = pipeline.query(query)
            print(result['answer'])
            print(f"\n[Retrieved from {len(result['sources'])} source(s)]")
        except Exception as e:
            print(f"Error: {e}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("6. Interactive Query Mode")
    print("=" * 60)
    print("Enter your questions (type 'quit' or 'exit' to stop):\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\nProcessing...")
            result = pipeline.query(question)
            
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\n[Retrieved from {len(result['sources'])} source(s)]")
            
            # Show source details
            if result['sources']:
                print("\nSources:")
                for j, source in enumerate(result['sources'], 1):
                    metadata = source.get('metadata', {})
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    table_name = metadata.get('table_name')
                    section = metadata.get('section')
                    
                    source_info = f"  {j}. Type: {chunk_type}"
                    if table_name:
                        source_info += f", Table: {table_name}"
                    if section:
                        source_info += f", Section: {section}"
                    if source.get('relevance_score'):
                        source_info += f", Relevance: {source['relevance_score']:.3f}"
                    print(source_info)
            
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

