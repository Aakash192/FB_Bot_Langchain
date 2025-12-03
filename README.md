# DOCX RAG Pipeline

A production-ready RAG (Retrieval-Augmented Generation) system for analyzing Word documents containing text and tabular data, with a focus on financial statement analysis.

## Features

- **Document Parsing**: Extract text and tables from Word documents (.docx)
- **Intelligent Chunking**: Smart text splitting with context preservation (1000 tokens per chunk)
- **Vector Embeddings**: OpenAI text-embedding-3-small for fast, accurate embeddings
- **Vector Database**: ChromaDB for persistent storage and efficient retrieval
- **RAG Query System**: GPT-4o-mini powered responses with top-3 relevant context retrieval
- **Table-Aware**: Special handling for tabular data with metadata preservation
- **Financial Analysis**: Optimized system prompts for financial document analysis

## Installation

### Quick Setup

Run the setup script to automatically install dependencies and verify your environment:

```bash
python setup.py
```

### Manual Setup

1. **Clone or download this repository**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root (you can copy `env_template.txt` to `.env`):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Project Structure

```
FB-Python/
├── docx_rag_pipeline.py    # Main pipeline implementation
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Place your .docx files here
└── chroma_db/             # Vector database storage (auto-created)
```

## Usage

### Basic Usage

```python
from docx_rag_pipeline import DocxExcelRagPipeline

# Initialize the pipeline
pipeline = DocxExcelRagPipeline(
    chroma_db_path="chroma_db",
    chunk_size=1000,
    retrieval_count=3,
    temperature=0.5
)

# Build vector store from documents
pipeline.build_vector_store(
    documents=["data/financial_report.docx"],
    reset=False  # Set to True to rebuild from scratch
)

# Query the documents
result = pipeline.query("What are the total revenues for Q1 2024?")
print(result['answer'])
print(f"\nRetrieved from {len(result['sources'])} sources")
```

### Advanced Usage

```python
# Parse multiple documents
documents = [
    "data/balance_sheet.docx",
    "data/income_statement.docx",
    "data/cash_flow.docx"
]

pipeline.build_vector_store(
    documents=documents,
    document_names=["Balance Sheet", "Income Statement", "Cash Flow"],
    reset=True
)

# Get collection statistics
stats = pipeline.get_collection_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Text chunks: {stats['text_chunks']}")
print(f"Table chunks: {stats['table_chunks']}")

# Query with detailed response
result = pipeline.query(
    "Compare the revenue trends across all quarters",
    include_metadata=True
)

print(result['answer'])
for i, source in enumerate(result['sources']):
    print(f"\nSource {i+1}:")
    print(f"  Relevance: {source['relevance_score']:.3f}")
    print(f"  Type: {source['metadata'].get('chunk_type')}")
    if source['metadata'].get('table_name'):
        print(f"  Table: {source['metadata']['table_name']}")
```

### Command Line Usage

Run the script directly to process documents in the `data/` folder:

```bash
python docx_rag_pipeline.py
```

## Configuration

### Pipeline Parameters

- `chunk_size`: Target chunk size in tokens (default: 1000)
- `retrieval_count`: Number of chunks to retrieve for queries (default: 3)
- `temperature`: LLM temperature for response generation (default: 0.5)
- `max_response_tokens`: Maximum tokens for LLM responses (default: 800)

### Models

- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)
- **LLM Model**: `gpt-4o-mini`

## System Prompt

The pipeline uses a specialized system prompt optimized for financial analysis:

- Always cites specific cell values, not approximations
- References table names and column headers
- Shows calculations step-by-step
- States data limitations clearly
- Compares data side-by-side when relevant
- Explains aggregation processes

## Document Parsing

The pipeline intelligently handles:

- **Text Sections**: Preserves document hierarchy and sections
- **Tables**: Extracts tables as separate chunks with:
  - Table names
  - Column headers
  - Row counts
  - Surrounding context
- **Metadata**: Tracks section names, document structure, and chunk relationships

## Error Handling

The pipeline includes robust error handling for:

- File parsing errors (graceful degradation)
- API rate limits (automatic retries with batching)
- Missing configuration (clear error messages)
- Network issues (fallback mechanisms)

## Performance Optimizations

- **Batch Embeddings**: Processes 50-100 chunks per API call
- **Lazy Loading**: Efficient handling of large documents
- **Persistent Storage**: ChromaDB for fast retrieval without re-indexing
- **Token Counting**: Efficient tiktoken-based chunking

## Example Queries

```python
# Numerical analysis
"What is the total revenue for 2024?"

# Comparison queries
"Compare Q1 and Q2 revenue figures"

# Aggregation queries
"What is the average profit margin across all quarters?"

# Table-specific queries
"What data is in Table_1? Show me the key metrics."

# Trend analysis
"Describe the revenue trend over the past 4 quarters"
```

## Requirements

- Python 3.9+ (recommended) or Python 3.8 with compatibility fixes
- OpenAI API key
- Internet connection for API calls

### Python 3.8 Compatibility

If you're using Python 3.8 and encounter `TypeError: 'type' object is not subscriptable` errors:

**Quick Fix (Windows PowerShell):**

```powershell
.\fix_compatibility.ps1
```

**Quick Fix (Linux/Mac):**

```bash
pip uninstall chromadb posthog
pip install 'chromadb==0.4.18' 'posthog<3.0.0' typing-extensions>=4.0.0
pip install -r requirements_python38.txt
```

**Option 1 (Recommended):** Upgrade to Python 3.9 or higher

**Option 2:** Manual fix:

```bash
pip uninstall chromadb
pip install 'chromadb==0.4.18' typing-extensions>=4.0.0
```

**Note:** The setup script automatically detects Python 3.8 and installs compatible versions.

## License

This project is provided as-is for production use.

## Support

For issues or questions, please check:

- OpenAI API documentation: https://platform.openai.com/docs
- ChromaDB documentation: https://docs.trychroma.com/
- python-docx documentation: https://python-docx.readthedocs.io/
