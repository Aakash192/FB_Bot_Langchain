"""
DOCX RAG Pipeline - Production-Ready RAG System for Document Analysis

This module provides a complete RAG (Retrieval-Augmented Generation) system
for analyzing Word documents containing text and tabular data, with a focus
on financial statement analysis.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Python 3.8 compatibility fix for ChromaDB/PostHog
# Set environment variables BEFORE importing chromadb
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = '1'

if sys.version_info < (3, 9):
    # Patch typing for Python 3.8 compatibility
    try:
        from typing_extensions import get_args, get_origin
    except ImportError:
        pass

import docx
from docx.document import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import openai
from openai import OpenAI

# Import chromadb after setting environment variables
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError as e:
    raise ImportError(
        f"Failed to import chromadb: {e}\n"
        "If you're using Python 3.8, try: pip install 'chromadb==0.4.18' typing-extensions>=4.0.0"
    ) from e

import tiktoken
from dotenv import load_dotenv

# Load environment variables (suppress warnings for comments in .env)
load_dotenv(verbose=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    chunk_id: str
    document_name: str
    chunk_type: str  # 'text' or 'table'
    table_name: Optional[str] = None
    table_headers: Optional[List[str]] = None
    row_count: Optional[int] = None
    section: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary for ChromaDB metadata storage.
        Filters out None values as ChromaDB doesn't accept them in metadata.
        Note: 'content' is excluded as it's stored separately in the 'documents' field.
        """
        metadata_dict = {
            'chunk_id': self.chunk_id,
            'document_name': self.document_name,
            'chunk_type': self.chunk_type,
        }
        
        # Add optional fields only if they are not None
        if self.table_name is not None:
            metadata_dict['table_name'] = self.table_name
        
        if self.table_headers is not None:
            metadata_dict['table_headers'] = json.dumps(self.table_headers)
        
        if self.row_count is not None:
            metadata_dict['row_count'] = self.row_count
        
        if self.section is not None:
            metadata_dict['section'] = self.section
        
        if self.page_number is not None:
            metadata_dict['page_number'] = self.page_number
        
        if self.metadata is not None:
            metadata_dict['metadata'] = json.dumps(self.metadata)
        
        return metadata_dict


class DocxExcelRagPipeline:
    """
    Production-ready RAG pipeline for analyzing Word documents with tabular data.
    
    Features:
    - Parse .docx files (text + tables)
    - Intelligent chunking with context preservation
    - OpenAI embeddings (text-embedding-3-small)
    - Chroma vector database storage
    - Retrieval-augmented generation with GPT-4o-mini
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        chroma_db_path: str = "chroma_db",
        collection_name: str = "docx_rag_collection",
        chunk_size: int = 1000,
        retrieval_count: int = 3,
        temperature: float = 0.5,
        max_response_tokens: int = 800
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            chroma_db_path: Path to ChromaDB storage directory
            collection_name: Name of the ChromaDB collection
            chunk_size: Target chunk size in tokens
            retrieval_count: Number of chunks to retrieve for queries
            temperature: LLM temperature for response generation
            max_response_tokens: Maximum tokens for LLM responses
        """
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-4o-mini"
        
        # Configuration
        self.chunk_size = chunk_size
        self.retrieval_count = retrieval_count
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        
        # Initialize tokenizer for chunking
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoding: {e}. Using fallback token counting.")
            self.tokenizer = None
        
        # Initialize ChromaDB
        self.chroma_db_path = Path(chroma_db_path)
        self.chroma_db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB (telemetry already disabled via environment variables)
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        except Exception as e:
            logger.warning(f"Error initializing ChromaDB with settings: {e}")
            # Fallback: try without explicit settings
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.chroma_db_path)
                )
            except Exception as e2:
                logger.error(f"Failed to initialize ChromaDB: {e2}")
                # Provide helpful error message for Python 3.8 compatibility issues
                if "not subscriptable" in str(e2) or "dict[str" in str(e2):
                    raise RuntimeError(
                        "ChromaDB compatibility issue detected. This may be due to Python 3.8. "
                        "Please try one of the following:\n"
                        "1. Upgrade to Python 3.9 or higher (recommended)\n"
                        "2. Install compatible ChromaDB version: pip install 'chromadb==0.4.18'\n"
                        "3. Install typing-extensions: pip install typing-extensions>=4.0.0"
                    ) from e2
                raise
        
        # Create embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=self.embedding_model
        )
        
        # Initialize collection
        self.collection_name = collection_name
        self.collection = None
        self._load_or_create_collection()
        
        # System prompt for financial analysis
        self.system_prompt = """You are an expert franchise analyst specializing in extracting accurate information from Franchise Disclosure Documents (FDDs).

KEY RULES:
- Extract and synthesize information from the provided context to answer questions comprehensively
- When answering about a specific franchise (e.g., Venture X, Snap-on Tools, WSI), prioritize information from that franchise's document
- Reference table names, column headers, and section names when providing answers
- If the question asks "what is X into" or "what does X do", look for business descriptions, services offered, industry information, or business model descriptions
- Synthesize information from multiple chunks if needed to provide a complete answer
- Be precise with numbers, dates, and financial figures when available
- If specific information is truly not available in the context, state what information IS available instead of just saying "not found"
- Always cite which document and section the information comes from

DOMAIN CONTEXT:
You analyze Franchise Disclosure Documents (FDDs) containing franchise information, financial data, and business details. These documents contain critical information about franchise opportunities. Extract relevant information from the provided context to answer questions thoroughly."""
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _load_or_create_collection(self):
        """Load existing collection or create a new one."""
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "DOCX RAG Collection for financial document analysis"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback method."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: approximate 1 token = 4 characters
        return len(text) // 4
    
    def parse_docx(self, file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Parse a Word document and extract text and tables.
        
        Args:
            file_path: Path to the .docx file
            
        Returns:
            Tuple of (text_chunks, tables) where:
            - text_chunks: List of text paragraphs/sections
            - tables: List of table dictionaries with data and metadata
        """
        try:
            doc = docx.Document(file_path)
            document_name = Path(file_path).stem
            
            text_chunks = []
            tables = []
            current_section = None
            current_text = []
            
            logger.info(f"Parsing document: {file_path}")
            
            # Process all elements in document order
            for element in doc.element.body:
                # Check if it's a paragraph
                if element.tag.endswith('p'):
                    para = Paragraph(element, doc)
                    text = para.text.strip()
                    
                    if not text:
                        continue
                    
                    # Check if it's a heading (section)
                    if para.style.name.startswith('Heading'):
                        # Save previous section if exists
                        if current_text:
                            text_chunks.append({
                                'text': '\n'.join(current_text),
                                'section': current_section
                            })
                            current_text = []
                        current_section = text
                        current_text.append(text)
                    else:
                        current_text.append(text)
                
                # Check if it's a table
                elif element.tag.endswith('tbl'):
                    # Save current text section before table
                    if current_text:
                        text_chunks.append({
                            'text': '\n'.join(current_text),
                            'section': current_section
                        })
                        current_text = []
                    
                    # Extract table
                    table = Table(element, doc)
                    table_data = self._extract_table_data(table)
                    if table_data:
                        tables.append({
                            'data': table_data,
                            'headers': table_data[0] if table_data else [],
                            'rows': table_data[1:] if len(table_data) > 1 else [],
                            'row_count': len(table_data) - 1 if len(table_data) > 1 else 0,
                            'column_count': len(table_data[0]) if table_data else 0,
                            'section': current_section,
                            'table_name': f"Table_{len(tables) + 1}"
                        })
            
            # Save remaining text
            if current_text:
                text_chunks.append({
                    'text': '\n'.join(current_text),
                    'section': current_section
                })
            
            logger.info(f"Extracted {len(text_chunks)} text sections and {len(tables)} tables")
            return text_chunks, tables
            
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            raise
    
    def _extract_table_data(self, table: Table) -> List[List[str]]:
        """
        Extract data from a Word table.
        
        Args:
            table: docx Table object
            
        Returns:
            List of rows, each row is a list of cell values
        """
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace('\n', ' ').replace('\r', ' ')
                row_data.append(cell_text)
            table_data.append(row_data)
        return table_data
    
    def parse_excel(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an Excel file and extract sheets and data.
        
        Note: This is a placeholder for Excel support. The current focus is on .docx files.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with sheet names as keys and data as values
        """
        try:
            import pandas as pd
            
            excel_data = {}
            xls = pd.ExcelFile(file_path)
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                excel_data[sheet_name] = {
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist(),
                    'row_count': len(df)
                }
            
            logger.info(f"Parsed Excel file: {file_path} with {len(excel_data)} sheets")
            return excel_data
            
        except ImportError:
            logger.warning("pandas and openpyxl required for Excel parsing. Install with: pip install pandas openpyxl")
            return {}
        except Exception as e:
            logger.error(f"Error parsing Excel file {file_path}: {e}")
            return {}
    
    def _format_table_for_chunk(self, table_data: Dict[str, Any], context: Optional[str] = None) -> str:
        """
        Format table data as a string for chunking.
        
        Args:
            table_data: Table dictionary with data, headers, etc.
            context: Optional surrounding context text
            
        Returns:
            Formatted string representation of the table
        """
        parts = []
        
        if context:
            parts.append(f"Context: {context}\n")
        
        if table_data.get('section'):
            parts.append(f"Section: {table_data['section']}\n")
        
        parts.append(f"Table: {table_data.get('table_name', 'Unnamed Table')}\n")
        parts.append("=" * 50 + "\n")
        
        # Add headers
        if table_data.get('headers'):
            headers = table_data['headers']
            parts.append(" | ".join(str(h) for h in headers))
            parts.append("\n" + "-" * 50 + "\n")
        
        # Add rows
        for row in table_data.get('rows', []):
            parts.append(" | ".join(str(cell) for cell in row))
            parts.append("\n")
        
        return "".join(parts)
    
    def chunk_and_enrich(
        self,
        text_chunks: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        document_name: str
    ) -> List[DocumentChunk]:
        """
        Split content into chunks and enrich with metadata.
        
        Args:
            text_chunks: List of text sections
            tables: List of table dictionaries
            document_name: Name of the source document
            
        Returns:
            List of DocumentChunk objects
        """
        all_chunks = []
        chunk_counter = 0
        
        # Process text chunks
        for text_data in text_chunks:
            text = text_data['text']
            section = text_data.get('section')
            
            # Split text into chunks of appropriate size
            text_chunks_split = self._split_text(text, self.chunk_size)
            
            for i, chunk_text in enumerate(text_chunks_split):
                chunk_id = f"{document_name}_text_{chunk_counter}"
                chunk = DocumentChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    document_name=document_name,
                    chunk_type='text',
                    section=section,
                    metadata={'chunk_index': i, 'total_chunks': len(text_chunks_split)}
                )
                all_chunks.append(chunk)
                chunk_counter += 1
        
        # Process tables as separate chunks
        for table_data in tables:
            # Get surrounding context (previous text section)
            context = None
            if table_data.get('section'):
                # Try to find related text
                for text_data in text_chunks:
                    if text_data.get('section') == table_data.get('section'):
                        context = text_data['text'][:500]  # First 500 chars for context
                        break
            
            # Format table
            table_content = self._format_table_for_chunk(table_data, context)
            
            chunk_id = f"{document_name}_table_{chunk_counter}"
            chunk = DocumentChunk(
                content=table_content,
                chunk_id=chunk_id,
                document_name=document_name,
                chunk_type='table',
                table_name=table_data.get('table_name'),
                table_headers=table_data.get('headers'),
                row_count=table_data.get('row_count'),
                section=table_data.get('section'),
                metadata={
                    'column_count': table_data.get('column_count'),
                    'table_index': len([c for c in all_chunks if c.chunk_type == 'table'])
                }
            )
            all_chunks.append(chunk)
            chunk_counter += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from document {document_name}")
        return all_chunks
    
    def _split_text(self, text: str, target_tokens: int) -> List[str]:
        """
        Split text into chunks of approximately target_tokens size.
        
        Args:
            text: Text to split
            target_tokens: Target number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Count tokens
        tokens = self._count_tokens(text)
        
        # If text fits in one chunk, return as is
        if tokens <= target_tokens:
            return [text]
        
        # Split by sentences first, then by paragraphs
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # If paragraph itself is too large, split by sentences
            if para_tokens > target_tokens:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split paragraph by sentences
                sentences = para.split('. ')
                for sentence in sentences:
                    sent_tokens = self._count_tokens(sentence)
                    if current_tokens + sent_tokens > target_tokens and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens
            else:
                # Check if adding this paragraph would exceed limit
                if current_tokens + para_tokens > target_tokens and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _batch_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Generate embeddings in batches to optimize API calls.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Retry with individual calls
                for text in batch:
                    try:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=[text]
                        )
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as retry_error:
                        logger.error(f"Error embedding individual text: {retry_error}")
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
        
        return all_embeddings
    
    def build_vector_store(
        self,
        documents: List[str],
        document_names: Optional[List[str]] = None,
        reset: bool = False
    ) -> None:
        """
        Build and persist the vector store from documents.
        
        Args:
            documents: List of file paths to .docx files
            document_names: Optional list of custom document names
            reset: If True, clear existing collection before adding
        """
        if reset:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass
            self._load_or_create_collection()
        
        all_chunks = []
        
        # Parse all documents
        for i, doc_path in enumerate(documents):
            doc_name = document_names[i] if document_names and i < len(document_names) else Path(doc_path).stem
            
            try:
                text_chunks, tables = self.parse_docx(doc_path)
                chunks = self.chunk_and_enrich(text_chunks, tables, doc_name)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
                continue
        
        if not all_chunks:
            logger.warning("No chunks to add to vector store")
            return
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in all_chunks]
        contents = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.to_dict() for chunk in all_chunks]
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._batch_embeddings(contents, batch_size=50)
        
        # Add to ChromaDB in batches to avoid batch size limits
        # ChromaDB has a maximum batch size (typically around 100-200)
        chroma_batch_size = 100
        total_chunks = len(all_chunks)
        chunks_added = 0
        
        try:
            for i in range(0, total_chunks, chroma_batch_size):
                batch_ids = ids[i:i + chroma_batch_size]
                batch_contents = contents[i:i + chroma_batch_size]
                batch_embeddings = embeddings[i:i + chroma_batch_size]
                batch_metadatas = metadatas[i:i + chroma_batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                chunks_added += len(batch_ids)
                logger.info(f"Added batch {i // chroma_batch_size + 1}/{(total_chunks - 1) // chroma_batch_size + 1}: {chunks_added}/{total_chunks} chunks")
            
            logger.info(f"Successfully added {chunks_added} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            logger.error(f"Failed at chunk {chunks_added}/{total_chunks}")
            raise
    
    def load_vector_store(self) -> bool:
        """
        Load the persisted vector store.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            count = self.collection.count()
            logger.info(f"Loaded vector store with {count} chunks")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def query(
        self,
        question: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a RAG query: retrieve relevant context and generate response.
        
        Args:
            question: User's question
            include_metadata: Whether to include metadata in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Enhance question for better semantic matching
            enhanced_query = question
            question_lower = question.lower()
            
            # Add context keywords for better retrieval
            if any(word in question_lower for word in ["what is", "what does", "into", "about", "business", "industry"]):
                # Add synonyms that might appear in documents
                enhanced_query = f"{question} business type industry services products operations"
            
            # Retrieve relevant chunks
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=self.retrieval_count
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    'answer': "No relevant information found in the documents.",
                    'sources': [],
                    'metadata': {}
                }
            
            # Prepare context from retrieved chunks
            retrieved_chunks = results['documents'][0]
            retrieved_metadatas = results['metadatas'][0] if results.get('metadatas') else []
            retrieved_distances = results['distances'][0] if results.get('distances') else []
            
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
                chunk_type = metadata.get('chunk_type', 'text')
                table_name = metadata.get('table_name')
                section = metadata.get('section')
                document_name = metadata.get('document_name', 'Unknown')
                
                context_header = f"\n--- Source {i+1} [Document: {document_name}] "
                if chunk_type == 'table' and table_name:
                    context_header += f"(Table: {table_name}) "
                if section:
                    context_header += f"[Section: {section}] "
                context_header += "---\n"
                
                context_parts.append(context_header + chunk)
            
            context = "\n".join(context_parts)
            
            # Generate response using LLM
            # Extract document names from metadata for better context
            document_names = set()
            for metadata in retrieved_metadatas:
                doc_name = metadata.get('document_name', '')
                if doc_name:
                    document_names.add(doc_name)
            
            document_context = ""
            if document_names:
                doc_list = ", ".join(sorted(document_names))
                document_context = f"\n\nAvailable documents in context: {doc_list}\n"
                # If question mentions a specific franchise, emphasize using only that document
                question_lower = question.lower()
                for doc_name in document_names:
                    doc_name_lower = doc_name.lower()
                    # Check if question mentions franchise name (partial match)
                    if any(word in doc_name_lower for word in question_lower.split() if len(word) > 3):
                        document_context += f"\nIMPORTANT: The question appears to be about '{doc_name}'. Only use information from this specific document.\n"
                        break
            
            # Enhance question understanding for better retrieval
            enhanced_question = question
            question_lower = question.lower()
            
            # Query expansion for common question patterns
            if "what is" in question_lower and ("into" in question_lower or "do" in question_lower or "about" in question_lower):
                # Add synonyms for business type questions
                enhanced_question = f"{question} What business or industry is this franchise in? What services or products does it offer?"
            elif "what does" in question_lower:
                enhanced_question = f"{question} What services, products, or business activities does this franchise provide?"
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"""Based on the following context from Franchise Disclosure Documents (FDDs), answer the question comprehensively by extracting relevant information.

INSTRUCTIONS:
- Extract and synthesize information from the provided context to answer the question
- If the question asks about what a franchise does or what business it's in, look for: business descriptions, services offered, products sold, industry type, business model, or operational details
- Use information from all relevant chunks in the context
- Cite specific values, table names, section names, and document names when available
- If you find partial information, provide what is available rather than saying nothing was found
- Synthesize information from multiple sources if needed for a complete answer

{document_context}
Context from documents:
{context}

Question: {question}

Provide a detailed, comprehensive answer by extracting relevant information from the context above. Include the document name and section/table when citing information."""
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_response_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Prepare response
            result = {
                'answer': answer,
                'sources': [
                    {
                        'content': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        'metadata': retrieved_metadatas[i] if i < len(retrieved_metadatas) else {},
                        'relevance_score': 1 - retrieved_distances[i] if i < len(retrieved_distances) else None
                    }
                    for i, chunk in enumerate(retrieved_chunks)
                ],
                'metadata': {
                    'model': self.llm_model,
                    'retrieval_count': len(retrieved_chunks),
                    'temperature': self.temperature
                }
            }
            
            logger.info(f"Generated response for query: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'metadata': {'error': str(e)}
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample to analyze
            sample = self.collection.peek(limit=min(100, count))
            metadatas = sample.get('metadatas', [])
            
            # Count by type
            text_count = sum(1 for m in metadatas if m.get('chunk_type') == 'text')
            table_count = sum(1 for m in metadatas if m.get('chunk_type') == 'table')
            
            # Get unique documents
            unique_docs = set(m.get('document_name') for m in metadatas if m.get('document_name'))
            
            return {
                'total_chunks': count,
                'text_chunks': text_count,
                'table_chunks': table_count,
                'unique_documents': len(unique_docs),
                'document_names': list(unique_docs)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


# Example usage and main function
if __name__ == "__main__":
    # Example usage
    pipeline = DocxExcelRagPipeline(
        chroma_db_path="chroma_db",
        chunk_size=1000,
        retrieval_count=3,
        temperature=0.5
    )
    
    # Build vector store from documents in data/ folder
    data_folder = Path("data")
    if data_folder.exists():
        docx_files = list(data_folder.glob("*.docx"))
        if docx_files:
            print(f"Found {len(docx_files)} .docx files")
            pipeline.build_vector_store(
                documents=[str(f) for f in docx_files],
                reset=False
            )
            
            # Get stats
            stats = pipeline.get_collection_stats()
            print(f"\nCollection Statistics:")
            print(json.dumps(stats, indent=2))
            
            # Example query
            print("\n" + "="*50)
            print("Example Query:")
            result = pipeline.query("What are the key financial metrics in the documents?")
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources: {len(result['sources'])} chunks retrieved")
        else:
            print("No .docx files found in data/ folder")
    else:
        print("data/ folder not found. Please create it and add .docx files.")

