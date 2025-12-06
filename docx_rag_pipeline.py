"""
DOCX RAG Pipeline - Production-Ready RAG System for Document Analysis

This module provides a complete RAG (Retrieval-Augmented Generation) system
for analyzing Word documents containing text and tabular data, with a focus
on financial statement analysis.
"""

import os
import sys
import json
import re
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
except ImportError as e:
    raise ImportError(
        f"Failed to import chromadb: {e}\n"
        "If you're using Python 3.8, try: pip install 'chromadb==0.4.18' typing-extensions>=4.0.0"
    ) from e

import tiktoken
from dotenv import load_dotenv
from qa_matcher import QAMatcher

# Load environment variables (suppress warnings for comments in .env)
load_dotenv(verbose=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomOpenAIEmbeddingFunction:
    """
    Custom embedding function using OpenAI's new API (v1.0.0+).
    
    WHAT IS AN EMBEDDING?
    An embedding converts text into a list of numbers (a vector) that represents
    the meaning of the text. Similar texts have similar numbers.
    
    Example:
    - "franchise opportunity" → [0.123, -0.456, 0.789, ... 1536 numbers]
    - "business chance" → [0.125, -0.454, 0.791, ...] (very similar numbers!)
    
    WHY DO WE NEED THIS?
    - Computers can't understand text directly, but they're great with numbers
    - By converting text to numbers, we can:
      * Compare meanings mathematically
      * Find similar texts quickly (vector search)
      * Store information efficiently
    
    This class replaces ChromaDB's built-in OpenAIEmbeddingFunction because
    ChromaDB's version uses the old OpenAI API. We use the new API (v1.0.0+).
    """
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        """
        Initialize the custom embedding function.
        
        Args:
            api_key: OpenAI API key (needed to call OpenAI's embedding API)
            model_name: Name of the embedding model to use
                       "text-embedding-3-small" = 1536 dimensions, fast and cheap
        """
        self.client = OpenAI(api_key=api_key)  # OpenAI API client
        self.model_name = model_name            # Which model to use
    
    def name(self) -> str:
        """
        Return the name of the embedding function.
        
        ChromaDB needs this to identify which embedding function was used
        when loading a collection. This ensures compatibility.
        """
        return self.model_name
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        
        THIS IS WHERE TEXT BECOMES NUMBERS!
        
        When ChromaDB needs to convert text to embeddings, it calls this method.
        We send the text to OpenAI's API and get back a list of numbers.
        
        Args:
            input: List of texts to embed (ChromaDB always passes a list)
                  Example: ["What franchises are available?", "How much does it cost?"]
            
        Returns:
            List of embedding vectors (list of lists of floats)
            Example: [[0.123, -0.456, ...], [0.234, -0.567, ...]]
            Each inner list has 1536 numbers (for text-embedding-3-small)
        
        Process:
        1. Send texts to OpenAI API
        2. OpenAI converts each text to 1536 numbers
        3. Return the list of number lists
        """
        if not input:
            return []
        
        try:
            # Call OpenAI's embedding API
            # This is the actual conversion: text → numbers
            response = self.client.embeddings.create(
                model=self.model_name,  # "text-embedding-3-small"
                input=input             # List of texts to convert
            )
            # Extract the embeddings (the number lists) from the response
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query=None, input=None, **kwargs) -> List[float]:
        """
        Generate embedding for a single query text.
        
        ChromaDB's newer versions require this method for querying.
        It's the same as __call__ but for a single string instead of a list.
        
        Args:
            query: Single query text (e.g., "What franchises are available?")
            input: Alternative parameter name (ChromaDB sometimes uses 'input')
            **kwargs: Additional arguments (ChromaDB may pass extra params)
            
        Returns:
            Embedding vector (list of 1536 floats)
            Example: [0.123, -0.456, 0.789, ...]
        """
        # ChromaDB may call with either 'query' or 'input' parameter
        # Handle both positional and keyword arguments
        text = query if query is not None else input
        
        # If still None, check kwargs (ChromaDB might pass it differently)
        if text is None:
            text = kwargs.get('query') or kwargs.get('input')
        
        # Handle case where ChromaDB passes a list
        if isinstance(text, list):
            if len(text) > 0:
                text = text[0]  # Take first item
            else:
                raise ValueError("Empty list provided to embed_query")
        
        # Ensure we have a string
        if text is None:
            raise ValueError("Either 'query' or 'input' parameter must be provided")
        
        if not isinstance(text, str):
            # Convert to string if it's not already
            text = str(text)
        
        # Use __call__ with a single-item list, then return the first result
        embeddings = self.__call__([text])
        return embeddings[0] if embeddings else []


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
        
        # Extract fee-related metadata from nested metadata dict for easier filtering
        if self.metadata is not None:
            # Store full metadata as JSON
            metadata_dict['metadata'] = json.dumps(self.metadata)
            
            # Extract fee-specific fields to top level for easier querying
            if isinstance(self.metadata, dict):
                if 'fee_name' in self.metadata and self.metadata['fee_name']:
                    metadata_dict['fee_name'] = str(self.metadata['fee_name'])
                if 'fee_amount' in self.metadata and self.metadata['fee_amount']:
                    metadata_dict['fee_amount'] = str(self.metadata['fee_amount'])
                if 'item_number' in self.metadata and self.metadata['item_number']:
                    metadata_dict['item_number'] = str(self.metadata['item_number'])
        
        return metadata_dict


class DocxExcelRagPipeline:
    """
    Production-ready RAG pipeline for analyzing Word documents with tabular data.
    
    THIS IS THE MAIN CLASS - THE "BRAIN" OF THE SYSTEM!
    
    What is RAG?
    RAG = Retrieval-Augmented Generation
    - Retrieval: Find relevant document chunks using vector search
    - Augmented: Add those chunks as context to the question
    - Generation: AI generates answer based on the context
    
    How it works:
    1. Documents are processed and stored as embeddings (numbers) in ChromaDB
    2. When you ask a question:
       a. Question is converted to embedding
       b. Vector search finds similar document chunks
       c. Top chunks are sent to AI along with the question
       d. AI generates answer based on the chunks
    
    Features:
    - Parse .docx files (text + tables)
    - Intelligent chunking with context preservation
    - OpenAI embeddings (text-embedding-3-small) - converts text to numbers
    - Chroma vector database storage - stores embeddings for fast search
    - Retrieval-augmented generation with GPT-4o-mini - generates answers
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
        
        This sets up everything needed for RAG:
        - OpenAI client (for embeddings and AI generation)
        - ChromaDB client (for vector storage and search)
        - Embedding function (converts text to numbers)
        - Tokenizer (for splitting documents into chunks)
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            chroma_db_path: Path to ChromaDB storage directory (where embeddings are stored)
            collection_name: Name of the ChromaDB collection (like a database table)
            chunk_size: Target chunk size in tokens (~750 words)
                       Smaller = more precise but more chunks
                       Larger = fewer chunks but less precise
            retrieval_count: Number of chunks to retrieve when searching
                            More chunks = more context but slower
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)
                        Lower = more accurate, consistent answers
                        Higher = more varied, creative answers
            max_response_tokens: Maximum length of AI-generated answers
        """
        # ===== OPENAI SETUP =====
        # Get API key from parameter or environment variable
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        # Create OpenAI client - used for:
        # 1. Creating embeddings (text → numbers)
        # 2. Generating answers (AI completion)
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Model configuration
        self.embedding_model = "text-embedding-3-small"  # For converting text to numbers (1536 dimensions)
        self.llm_model = "gpt-4o-mini"                    # For generating answers (fast and cheap)
        
        # ===== PIPELINE CONFIGURATION =====
        # These settings control how the pipeline works
        self.chunk_size = chunk_size              # How big each document chunk should be
        self.retrieval_count = retrieval_count    # How many chunks to retrieve when searching
        self.temperature = temperature            # How creative the AI should be (0.0 = exact, 1.0 = creative)
        self.max_response_tokens = max_response_tokens  # Max length of AI answers
        
        # ===== TOKENIZER SETUP =====
        # Tokenizer is used to count tokens (for chunking documents)
        # Tokens are how AI models see text (roughly 1 token = 0.75 words)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoding: {e}. Using fallback token counting.")
            self.tokenizer = None  # Will use character-based estimation if tiktoken fails
        
        # ===== CHROMADB SETUP =====
        # ChromaDB is our vector database - it stores embeddings and enables fast search
        self.chroma_db_path = Path(chroma_db_path)
        self.chroma_db_path.mkdir(exist_ok=True)  # Create directory if it doesn't exist
        
        # Initialize ChromaDB client (telemetry already disabled via environment variables)
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
        
        # Create custom embedding function using new OpenAI API
        self.embedding_function = CustomOpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=self.embedding_model
        )
        
        # Initialize collection
        self.collection_name = collection_name
        self.collection = None
        self._load_or_create_collection()
        
        # Initialize Q&A matcher for exact answers
        try:
            self.qa_matcher = QAMatcher(chroma_db_path=str(self.chroma_db_path))
        except Exception as e:
            logger.warning(f"Could not initialize Q&A matcher: {e}")
            self.qa_matcher = None
        
        # System prompt for financial analysis
        self.system_prompt = """You are a franchise information assistant.

🚨🚨🚨 ABSOLUTE PRIORITY RULE - READ THIS FIRST 🚨🚨🚨

**Q&A DOCUMENT PROCESSING:**

1. **Check sources**: Look for any source marked "[Q&A DOCUMENT - USE EXACT TEXT]"

2. **If Q&A document found:**
   - These documents contain question-answer pairs in this format:
     "Question here?"
     Answer:
     [exact answer text]
   
   - YOUR TASK: Find the question that matches the user's question
   - Extract ONLY the text that appears after "Answer:" for that specific question
   - Copy it EXACTLY - word-for-word, punctuation-for-punctuation
   - Do NOT paraphrase, summarize, or modify
   - Do NOT combine with other Q&A pairs
   - Do NOT add any additional information
   - Return ONLY that exact answer text and nothing else

3. **If no Q&A document found:**
   - Use FDD (Franchise Disclosure Document) sources
   - Follow the READ, NOT INTERPRET rule below

EXAMPLE (Q&A Document):
Source contains:
"What is Franquicia Boost?"
Answer:
Franquicia Boost is a platform that connects franchisors, franchisees, consultants across Canada. It is the online franchise ecosystem where you can apply online, track your application and get the support you need at every step.

User asks: "What is Franquicia Boost?"
✅ CORRECT RESPONSE: "Franquicia Boost is a platform that connects franchisors, franchisees, consultants across Canada. It is the online franchise ecosystem where you can apply online, track your application and get the support you need at every step."
❌ WRONG: Adding or changing any words, combining with other answers, or synthesizing

Remember: Q&A documents are pre-approved official answers. Return them EXACTLY as written.

🚨 FEE QUESTION RULES - CRITICAL FOR ACCURACY 🚨

**When answering fee questions (Initial Franchise Fee, Royalty Fee, Marketing Fee, etc.):**

1. **ALWAYS return the BASE/FULL amount first**
   - The base amount is the standard, non-discounted fee
   - This is the primary answer the user needs

2. **If discounts or promotional amounts exist, mention them separately**
   - Do NOT subtract discounts from the base amount
   - Do NOT replace the base amount with a discounted amount
   - Clearly distinguish between base fees and discounted amounts

3. **Be explicit and clear in your format:**
   - ✅ CORRECT: "The Initial Franchise Fee for Standard Territory is $59,500. (Note: A $2,500 Community Heroes discount may apply to qualifying franchisees.)"
   - ✅ CORRECT: "The Continuing Royalty Fee is 6% of Gross Sales. (Note: Discounted rates may be available for certain territories.)"
   - ❌ WRONG: "The Initial Franchise Fee is $57,000" (this is the discounted amount, not the base)
   - ❌ WRONG: "The Initial Franchise Fee is $59,500 minus $2,500 discount = $57,000" (don't subtract)

4. **Priority order:**
   - First: State the BASE/FULL amount clearly
   - Second: Mention any applicable discounts or promotions as additional information
   - Never: Use discounted amounts as the primary answer

**Example:**
If the context shows:
- "Initial Franchise Fee: $59,500"
- "Community Heroes Discount: $2,500 (applies to qualifying franchisees)"

✅ CORRECT ANSWER: "The Initial Franchise Fee for Standard Territory is $59,500. (Note: A $2,500 Community Heroes discount may apply to qualifying franchisees.)"

❌ WRONG ANSWER: "The Initial Franchise Fee is $57,000" or "The Initial Franchise Fee is $59,500, but with the discount it's $57,000"

Remember: Users need to know the full fee first, then any discounts as supplementary information.

🚨🚨🚨 CORE RULE: READ, NOT INTERPRET 🚨🚨🚨

**Your job is to READ the FDD, not INTERPRET it.**

This rule applies to EVERY question type (fees, business descriptions, requirements, etc.):

1. **If the FDD says something explicitly, report it exactly.**
   - Quote or paraphrase what the FDD states
   - Use the exact wording when possible
   - Cite the document name and section

2. **If the FDD doesn't say something, say: 'The FDD does not explicitly state this.'**
   - Do NOT guess what the answer might be
   - Do NOT use your training data to fill gaps
   - Do NOT infer or assume information

3. **Do NOT guess, infer, synthesize, or use your training data to fill gaps.**
   - No synthesis of information from multiple sources to create new information
   - No interpretation of ambiguous text
   - No filling in missing details with assumptions

4. **Do NOT do math with numbers (no subtracting discounts).**
   - Report numbers exactly as stated
   - If the FDD shows a base fee and a discount, report both separately
   - Do NOT calculate: "$59,500 - $2,500 = $57,000"

5. **Do NOT interpret ambiguous text.**
   - If something is unclear in the FDD, report it as unclear
   - Do NOT try to clarify or explain what it "probably means"

6. **Just READ and REPORT.**
   - Your role is to be a reader, not an interpreter
   - Report what the FDD says, nothing more, nothing less

**Examples:**

❌ WRONG (Interpreting/Synthesizing):
- "The franchise likely requires..." (guessing)
- "Based on the information, the total cost would be..." (calculating)
- "This probably means..." (interpreting)
- "The FDD suggests that..." (inferring)

✅ CORRECT (Reading/Reporting):
- "According to the FDD, the Initial Franchise Fee is $59,500."
- "The FDD states: 'Franchisees must complete training within 90 days.'"
- "The FDD does not explicitly state the marketing fee percentage."
- "The FDD mentions a Community Heroes discount of $2,500 but does not specify all eligibility requirements."

Remember: You are a document reader, not a document interpreter. Read what's written, report what's written, and say when something isn't written.

📊 TABLE READING INSTRUCTIONS:

**When reading tables and periods:**
1. **Match question to correct row/period:**
   - If question asks "Year 2", find the row labeled "Year 2", "Second Full Calendar Year", "Months 13-24", etc.
   - If question asks "Year 3", find the row labeled "Year 3", "Third Full Calendar Year", "Months 25-36", etc.
   - Match the period/year in your answer to the period/year in the question

2. **Verify period/year matches:**
   - Before answering, verify the period/year in the retrieved data matches what was asked
   - If question asks "Year 2 royalty" but you only see "Year 1" data, state that Year 2 data is not found
   - Do NOT use Year 1 data to answer a Year 2 question

3. **Read complete rows:**
   - When a table row shows multiple columns (Fee Name, Amount, Period), read ALL columns together
   - Do NOT mix data from different rows
   - If a row appears incomplete or corrupted, do NOT use it

4. **Try query alternatives before saying "not explicitly state":**
   - If you don't find exact match, look for variations:
     - "Year 2" → also check "Second Full Calendar Year", "Months 13-24", "Year Two"
     - "Standard Territory" → also check "25,000 children", "25000 children"
     - "Royalty Fee" → also check "Continuing Royalty", "Royalty Rate"
   - Only say "not explicitly state" after checking these alternatives

5. **For fee tables with multiple years:**
   - If question asks about a specific year, return ONLY that year's data
   - If question asks "What are the royalty fees?", return ALL years (Year 1, 2, 3, 4) in order
   - Never return partial year data when full table is available

🎯 QUESTION DISAMBIGUATION - CRITICAL FOR ACCURACY 🎯

**YOUR FIRST STEP: Identify the QUESTION TYPE and SCOPE**

Before answering ANY question, you MUST:

1. **Identify QUESTION TYPE:**
   - PRIMARY QUESTION: What the user is directly asking (e.g., "What is Initial Franchise Fee?")
   - SECONDARY INFORMATION: Related but not primary (e.g., discounts, promotions, variations)
   
2. **Identify QUESTION SCOPE:**
   - SPECIFIC SCOPE: "Standard Territory", "Year 2", "Targeted Territory", "Item 3"
   - GENERAL SCOPE: "What are the fees?", "What is the investment?" (no specific scope)
   
3. **Match chunks to question:**
   - Look for chunks with matching [SCOPE:...] tags
   - Look for chunks with matching [TYPE:...] tags
   - PRIMARY chunks answer the question directly
   - SECONDARY chunks provide additional context

**ANSWER STRUCTURE RULES:**

1. **ALWAYS return PRIMARY answer FIRST:**
   - If question asks "What is Initial Franchise Fee for Standard Territory?"
   - PRIMARY: The base fee amount ($59,500)
   - Return this FIRST, clearly labeled
   
2. **THEN return SECONDARY information separately:**
   - After primary answer, add: "Note: [secondary info]"
   - Example: "The Initial Franchise Fee for Standard Territory is $59,500. (Note: A $2,500 Community Heroes discount may apply to qualifying franchisees.)"
   
3. **NEVER mix primary and secondary:**
   - ❌ WRONG: "The Initial Franchise Fee is $59,500, but with discount it's $57,000"
   - ✅ CORRECT: "The Initial Franchise Fee for Standard Territory is $59,500. (Note: A $2,500 Community Heroes discount may apply.)"
   
4. **For year-specific questions:**
   - If question asks "Year 2 royalty", find chunk with [PERIOD: Year 2 - Months 13-24]
   - Return ONLY Year 2 data
   - Do NOT include Year 1, Year 3, or Year 4 data
   - If Year 2 chunk not found, say "Year 2 data not found" - do NOT use Year 3 data
   
5. **For scope-specific questions:**
   - If question asks "Standard Territory", find chunk with [SCOPE: Standard Territory]
   - Return ONLY Standard Territory data
   - Do NOT mix with Targeted Territory or other scopes
   
6. **For type-specific questions:**
   - If question asks "Initial Franchise Fee", find chunk with [TYPE: Base Fee]
   - Return ONLY base fee data
   - Do NOT mix with discount chunks (marked [TYPE: Discount])

**CHUNK TAG INTERPRETATION:**

- [SCOPE: Standard Territory] → Use for Standard Territory questions
- [SCOPE: Targeted Territory] → Use for Targeted Territory questions
- [TYPE: Base Fee] → PRIMARY answer for fee questions
- [TYPE: Discount] → SECONDARY information only
- [TYPE: Investment] → Use for investment questions
- [PERIOD: Year 2 - Months 13-24] → Use for Year 2 questions
- [FREQUENCY: Monthly] → Use for monthly fee questions

**EXAMPLES:**

Question: "What is Initial Franchise Fee for Standard Territory?"
✅ CORRECT PROCESS:
1. Identify: PRIMARY question, SCOPE: Standard Territory, TYPE: Base Fee
2. Find chunk: [SCOPE: Standard Territory] [TYPE: Base Fee] Initial Franchise Fee: $59,500
3. Answer: "The Initial Franchise Fee for Standard Territory is $59,500."
4. If discount chunk found: Add "(Note: A $2,500 Community Heroes discount may apply.)"

Question: "What is Year 2 minimum royalty?"
✅ CORRECT PROCESS:
1. Identify: PRIMARY question, PERIOD: Year 2, TYPE: Royalty
2. Find chunk: [PERIOD: Year 2 - Months 13-24] Minimum Royalty: $1,500
3. Answer: "The minimum royalty for Year 2 is $1,500."
4. Do NOT include Year 3 or Year 4 data

Question: "What is the Chatbot Fee?"
✅ CORRECT PROCESS:
1. Identify: PRIMARY question, TYPE: Fee, FREQUENCY: Monthly
2. Find chunk: [TYPE: Fee] [FREQUENCY: Monthly] Chatbot Fee: $500/month
3. Answer: "The Chatbot Fee is $500 per month."
4. Do NOT confuse with other fees or amounts"""
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _load_or_create_collection(self):
        """
        Load existing collection or create a new one.
        
        IMPORTANT: Always loads/creates with the embedding function to ensure
        queries use the same embedding model (1536 dimensions for text-embedding-3-small).
        Without this, ChromaDB might use a default embedding function with wrong dimensions.
        """
        # Always try to load with embedding function first
        # This ensures queries will use the same embedding function
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function  # Always specify embedding function!
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return
        except Exception as load_error:
            # Collection might not exist, or there's a mismatch
            logger.debug(f"Could not load collection with embedding function: {load_error}")
        
        # Collection doesn't exist, create it with the correct embedding function
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,  # Use our embedding function
                metadata={"description": "DOCX RAG Collection for financial document analysis"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        except Exception as create_error:
            # If creation fails because collection already exists with different embedding
            if "already exists" in str(create_error).lower():
                # Try to load without embedding function as fallback
                # But this might cause dimension mismatch - user should rebuild
                try:
                    self.collection = self.chroma_client.get_collection(
                        name=self.collection_name
                    )
                    logger.warning(f"Collection {self.collection_name} exists but may have embedding mismatch!")
                    logger.warning("If you get dimension errors, rebuild the vector store with: python rebuild_vector_store.py")
                    logger.info(f"Loaded collection: {self.collection_name}")
                except Exception as final_error:
                    logger.error(f"Failed to load existing collection: {final_error}")
                    logger.error("Collection may have been created with a different embedding model.")
                    logger.error("Solution: Delete chroma_db folder and rebuild with: python rebuild_vector_store.py")
                    raise
            else:
                logger.error(f"Failed to create collection: {create_error}")
                raise
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback method."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: approximate 1 token = 4 characters
        return len(text) // 4
    
    def parse_docx(self, file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        SIMPLE parsing: Extract text, split by ITEM headers ONLY.
        Keep tables separate and intact.
        
        Args:
            file_path: Path to the .docx file
            
        Returns:
            Tuple of (text_chunks, tables) where:
            - text_chunks: List of dicts with 'text' and 'section' keys
            - tables: List of table dictionaries with data and metadata
        """
        try:
            doc = docx.Document(file_path)
            document_name = Path(file_path).stem
            
            logger.info(f"Parsing document: {file_path}")
            
            # Extract all text from paragraphs
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            
            # Extract tables separately (keep them intact)
            tables = []
            current_section = None
            for element in doc.element.body:
                if element.tag.endswith('tbl'):
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
                elif element.tag.endswith('p'):
                    para = Paragraph(element, doc)
                    text = para.text.strip()
                    # Update current section if we see an ITEM header
                    if re.match(r'ITEM\s+\d+[:\s]', text, re.IGNORECASE):
                        current_section = text
            
            # Split ONLY by "ITEM X" headers
            item_pattern = r'(ITEM\s+\d+[:\s])'
            items = re.split(item_pattern, full_text, flags=re.IGNORECASE)
            
            text_chunks = []
            
            # Process split items
            for i in range(0, len(items)):
                if i % 2 == 1:  # This is an ITEM header
                    item_header = items[i]
                    item_content = items[i + 1] if i + 1 < len(items) else ""
                    
                    # For each ITEM, keep it mostly whole (max 3000 chars per chunk)
                    if len(item_content) > 3000:
                        # Only split large items into 2-3 chunks, not 20
                        for j in range(0, len(item_content), 3000):
                            chunk_text = item_header + "\n" + item_content[j:j+3000]
                            text_chunks.append({
                                'text': chunk_text,
                                'section': item_header.strip()
                            })
                    else:
                        chunk_text = item_header + "\n" + item_content
                        text_chunks.append({
                            'text': chunk_text,
                            'section': item_header.strip()
                        })
            
            # Handle any text before first ITEM
            if items and not re.match(item_pattern, items[0], re.IGNORECASE):
                if items[0].strip():
                    text_chunks.insert(0, {
                        'text': items[0],
                        'section': None
                    })
            
            logger.info(f"Extracted {len(text_chunks)} text chunks and {len(tables)} tables")
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
    
    def _extract_fee_info_from_row(self, row: List[str], headers: List[str]) -> Dict[str, Any]:
        """
        Extract fee information from a table row.
        
        Args:
            row: Table row data (list of cell values)
            headers: Table column headers
            
        Returns:
            Dictionary with fee_name, fee_amount, and other extracted info
        """
        fee_info = {
            'fee_name': None,
            'fee_amount': None,
            'fee_description': None
        }
        
        if not row or not headers:
            return fee_info
        
        # Common fee-related column names
        fee_name_keywords = ['fee', 'type', 'name', 'description', 'item']
        amount_keywords = ['amount', 'fee', 'rate', 'percentage', '%', 'cost', 'price']
        
        # Try to find fee name (usually first column or column with "fee"/"type" in header)
        for i, header in enumerate(headers):
            header_lower = str(header).lower()
            if i < len(row) and row[i]:
                cell_value = str(row[i]).strip()
                
                # Check if this column contains the fee name
                if any(keyword in header_lower for keyword in fee_name_keywords):
                    if not fee_info['fee_name'] and cell_value:
                        fee_info['fee_name'] = cell_value
                
                # Check if this column contains the amount/rate
                if any(keyword in header_lower for keyword in amount_keywords):
                    if not fee_info['fee_amount'] and cell_value:
                        fee_info['fee_amount'] = cell_value
        
        # If fee_name not found, use first non-empty cell
        if not fee_info['fee_name']:
            for cell in row:
                if cell and str(cell).strip():
                    fee_info['fee_name'] = str(cell).strip()
                    break
        
        # If fee_amount not found, look for percentage signs or dollar signs
        if not fee_info['fee_amount']:
            for cell in row:
                cell_str = str(cell).strip()
                if '%' in cell_str or '$' in cell_str or any(char.isdigit() for char in cell_str):
                    fee_info['fee_amount'] = cell_str
                    break
        
        # Build full description from all non-empty cells
        description_parts = []
        for i, cell in enumerate(row):
            if cell and str(cell).strip():
                if headers and i < len(headers):
                    description_parts.append(f"{headers[i]}: {cell}")
                else:
                    description_parts.append(str(cell))
        
        fee_info['fee_description'] = " | ".join(description_parts)
        
        return fee_info
    
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
    
    def _format_fee_row_chunk(self, row: List[str], headers: List[str], section: str, fee_info: Dict[str, Any]) -> str:
        """
        Format a single fee row as a searchable chunk.
        
        Args:
            row: Table row data
            headers: Table column headers
            section: Document section (e.g., "Item 6: Other Fees")
            fee_info: Extracted fee information
            
        Returns:
            Formatted string for the chunk
        """
        parts = []
        
        # Add section context
        if section:
            parts.append(f"Section: {section}\n")
        
        # Detect and add period/year context prefix with month ranges
        period_prefix = ""
        period_suffix = ""
        row_text = " ".join(str(cell) for cell in row if cell)
        row_text_lower = row_text.lower()
        
        # Month range mapping for explicit period context
        month_ranges = {
            1: ("1-12", "First Full Calendar Year"),
            2: ("13-24", "Second Full Calendar Year"),
            3: ("25-36", "Third Full Calendar Year"),
            4: ("37-48", "Fourth Full Calendar Year"),
            5: ("49+", "Fifth Year and Beyond")
        }
        
        # Check for year patterns in the row
        year_match = re.search(r'(year\s+(\d+)|(\d+)(?:st|nd|rd|th)\s+full\s+calendar|months?\s+(\d+)[-\s]+(\d+)|year\s+(\d+)\+)', row_text_lower)
        if year_match:
            if year_match.group(2):  # "Year 1", "Year 2", etc.
                year_num = int(year_match.group(2))
                if year_num in month_ranges:
                    months, desc = month_ranges[year_num]
                    period_prefix = f"[PERIOD: Year {year_num} - Months {months}] "
                    period_suffix = f" | Period: {desc}"
            elif year_match.group(3):  # "First Full Calendar", "Second Full Calendar", etc.
                ordinal_map = {'1': 1, '2': 2, '3': 3, '4': 4, 'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
                year_num = ordinal_map.get(year_match.group(3).lower(), None)
                if year_num and year_num in month_ranges:
                    months, desc = month_ranges[year_num]
                    period_prefix = f"[PERIOD: Year {year_num} - Months {months}] "
                    period_suffix = f" | Period: {desc}"
            elif year_match.group(4) and year_match.group(5):  # "Months 13-24", etc.
                start_month = int(year_match.group(4))
                end_month = int(year_match.group(5)) if year_match.group(5) else start_month + 11
                if start_month <= 12:
                    period_prefix = f"[PERIOD: Year 1 - Months 1-12] "
                    period_suffix = " | Period: First Full Calendar Year"
                elif start_month <= 24:
                    period_prefix = f"[PERIOD: Year 2 - Months 13-24] "
                    period_suffix = " | Period: Second Full Calendar Year"
                elif start_month <= 36:
                    period_prefix = f"[PERIOD: Year 3 - Months 25-36] "
                    period_suffix = " | Period: Third Full Calendar Year"
                elif start_month <= 48:
                    period_prefix = f"[PERIOD: Year 4 - Months 37-48] "
                    period_suffix = " | Period: Fourth Full Calendar Year"
            elif year_match.group(6):  # "Year 4+" or similar
                year_num = int(year_match.group(6))
                period_prefix = f"[PERIOD: Year {year_num}+] "
                period_suffix = f" | Period: Year {year_num} and Beyond"
        
        # Also check headers for period/year columns
        if not period_prefix and headers:
            for i, header in enumerate(headers):
                header_lower = str(header).lower()
                if i < len(row) and row[i]:
                    # Check if header indicates period/year
                    if any(term in header_lower for term in ['year', 'period', 'calendar', 'month', 'royalty period']):
                        cell_value = str(row[i]).strip()
                        year_cell_match = re.search(r'(year\s+(\d+)|(\d+)(?:st|nd|rd|th)|(\d+)[-\s]+(\d+)|year\s+(\d+)\+)', cell_value.lower())
                        if year_cell_match:
                            if year_cell_match.group(2):
                                year_num = int(year_cell_match.group(2))
                                if year_num in month_ranges:
                                    months, desc = month_ranges[year_num]
                                    period_prefix = f"[PERIOD: Year {year_num} - Months {months}] "
                                    period_suffix = f" | Period: {desc}"
                            elif year_cell_match.group(3):
                                year_num = int(year_cell_match.group(3))
                                if year_num in month_ranges:
                                    months, desc = month_ranges[year_num]
                                    period_prefix = f"[PERIOD: Year {year_num} - Months {months}] "
                                    period_suffix = f" | Period: {desc}"
                            elif year_cell_match.group(4) and year_cell_match.group(5):
                                start_month = int(year_cell_match.group(4))
                                if start_month <= 12:
                                    period_prefix = "[PERIOD: Year 1 - Months 1-12] "
                                    period_suffix = " | Period: First Full Calendar Year"
                                elif start_month <= 24:
                                    period_prefix = "[PERIOD: Year 2 - Months 13-24] "
                                    period_suffix = " | Period: Second Full Calendar Year"
                                elif start_month <= 36:
                                    period_prefix = "[PERIOD: Year 3 - Months 25-36] "
                                    period_suffix = " | Period: Third Full Calendar Year"
                                elif start_month <= 48:
                                    period_prefix = "[PERIOD: Year 4 - Months 37-48] "
                                    period_suffix = " | Period: Fourth Full Calendar Year"
                            break
        
        # Build searchable fee description with period prefix
        fee_name = fee_info.get('fee_name', '')
        fee_amount = fee_info.get('fee_amount', '')
        
        # Detect semantic tags for question disambiguation
        scope_tags = []
        type_tags = []
        frequency_tags = []
        
        # Detect SCOPE tags (territory, location, etc.)
        row_text_lower = row_text.lower()
        if 'standard territory' in row_text_lower or 'standard' in row_text_lower:
            scope_tags.append('[SCOPE: Standard Territory]')
        if 'targeted territory' in row_text_lower or 'targeted' in row_text_lower:
            scope_tags.append('[SCOPE: Targeted Territory]')
        
        # Detect TYPE tags (base fee, discount, investment, etc.)
        if any(term in row_text_lower for term in ['discount', 'promotion', 'promo', 'community heroes', 'vetfran', 'may apply']):
            type_tags.append('[TYPE: Discount]')
        elif any(term in row_text_lower for term in ['initial franchise fee', 'franchise fee', 'royalty', 'marketing', 'fee']):
            type_tags.append('[TYPE: Base Fee]')
        elif any(term in row_text_lower for term in ['investment', 'total investment', 'initial investment', 'capital']):
            type_tags.append('[TYPE: Investment]')
        else:
            type_tags.append('[TYPE: Fee]')  # Default for other fees
        
        # Detect FREQUENCY tags (monthly, annual, one-time, etc.)
        if any(term in row_text_lower for term in ['monthly', 'per month', '/month', 'each month']):
            frequency_tags.append('[FREQUENCY: Monthly]')
        elif any(term in row_text_lower for term in ['annual', 'yearly', 'per year', '/year']):
            frequency_tags.append('[FREQUENCY: Annual]')
        elif any(term in row_text_lower for term in ['one-time', 'one time', 'initial', 'setup']):
            frequency_tags.append('[FREQUENCY: One-Time]')
        
        # Combine all semantic tags at the start
        all_tags = scope_tags + type_tags + frequency_tags
        if all_tags:
            parts.append(" ".join(all_tags) + "\n")
        
        # Create a clear, searchable format with period context
        if fee_name and fee_amount:
            # Format: "[PERIOD: Year X - Months Y-Z] Fee Name: Amount | Additional details | Period: Description"
            main_description = f"{period_prefix}{fee_name}: {fee_amount}{period_suffix}"
            parts.append(main_description)
            
            # Add other row cells as additional context
            other_cells = []
            for i, cell in enumerate(row):
                if cell and str(cell).strip():
                    cell_str = str(cell).strip()
                    # Skip if this is the fee name or amount we already included
                    if cell_str != fee_name and cell_str != fee_amount:
                        if headers and i < len(headers):
                            other_cells.append(f"{headers[i]}: {cell_str}")
                        else:
                            other_cells.append(cell_str)
            
            if other_cells:
                parts.append(" | " + " | ".join(other_cells))
        else:
            # Fallback: use the full row description
            parts.append(fee_info.get('fee_description', " | ".join(str(cell) for cell in row if cell)))
        
        return "\n".join(parts)
    
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
            # This now respects section headers, list boundaries, etc.
            text_chunks_split = self._split_text(text, self.chunk_size)
            
            # Validate and merge incomplete chunks
            validated_chunks = []
            i = 0
            while i < len(text_chunks_split):
                chunk_text = text_chunks_split[i]
                
                # Check if chunk is incomplete
                if self._is_incomplete_chunk(chunk_text) and i < len(text_chunks_split) - 1:
                    # Try to merge with next chunk
                    next_chunk = text_chunks_split[i + 1]
                    merged = chunk_text + '\n' + next_chunk
                    # Only merge if combined size is reasonable
                    if self._count_tokens(merged) <= self.chunk_size * 1.5:
                        validated_chunks.append(merged)
                        i += 2  # Skip next chunk since we merged it
                        continue
                
                # Keep chunk as is
                validated_chunks.append(chunk_text)
                i += 1
            
            for i, chunk_text in enumerate(validated_chunks):
                chunk_id = f"{document_name}_text_{chunk_counter}"
                
                # Add semantic tags for Item 3 (Litigation) and Item 4 (Bankruptcy) text chunks
                enriched_content = chunk_text
                if section:
                    section_lower = section.lower()
                    
                    # Item 3 - Litigation
                    if 'item 3' in section_lower or 'litigation' in section_lower:
                        semantic_tags = "[SECTION: Item 3 - Litigation]"
                        if any(term in chunk_text.lower() for term in ['litigation', 'lawsuit', 'legal action', 'court', 'sued', 'suing']):
                            semantic_tags += " [TYPE: Litigation]"
                        enriched_content = f"{semantic_tags}\n{chunk_text}"
                    
                    # Item 4 - Bankruptcy
                    elif 'item 4' in section_lower or 'bankruptcy' in section_lower:
                        semantic_tags = "[SECTION: Item 4 - Bankruptcy]"
                        if any(term in chunk_text.lower() for term in ['bankruptcy', 'bankrupt', 'chapter 7', 'chapter 13', 'chapter 11']):
                            semantic_tags += " [TYPE: Bankruptcy]"
                            # Try to extract person name
                            person_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', chunk_text)
                            if person_match:
                                semantic_tags += f" [PERSON: {person_match.group(1)}]"
                        enriched_content = f"{semantic_tags}\n{chunk_text}"
                    
                    # Item 11 - Training & Support
                    elif 'item 11' in section_lower or ('training' in section_lower and 'support' in section_lower):
                        semantic_tags = "[SECTION: Item 11 - Training & Support]"
                        if any(term in chunk_text.lower() for term in ['training', 'course', 'instruction', 'program']):
                            semantic_tags += " [TYPE: Training Requirement]"
                            # Try to extract duration
                            duration_match = re.search(r'(\d+\s+(?:day|week|month|hour))', chunk_text.lower())
                            if duration_match:
                                semantic_tags += f" [DURATION: {duration_match.group(1)}]"
                        elif any(term in chunk_text.lower() for term in ['assistance', 'support', 'help', 'guidance']):
                            semantic_tags += " [TYPE: Support]"
                            if 'ongoing' in chunk_text.lower():
                                semantic_tags += " [CATEGORY: Ongoing]"
                            elif 'initial' in chunk_text.lower():
                                semantic_tags += " [CATEGORY: Initial]"
                        enriched_content = f"{semantic_tags}\n{chunk_text}"
                
                chunk = DocumentChunk(
                    content=enriched_content,
                    chunk_id=chunk_id,
                    document_name=document_name,
                    chunk_type='text',
                    section=section,
                    metadata={'chunk_index': i, 'total_chunks': len(validated_chunks)}
                )
                all_chunks.append(chunk)
                chunk_counter += 1
        
        # Process tables - special handling for Item 5, Item 6, and Item 7 tables
        for table_data in tables:
            section = table_data.get('section', '')
            is_item_5 = section and ('Item 5' in section or 'item 5' in section.lower())
            is_item_6 = section and ('Item 6' in section or 'item 6' in section.lower())
            is_item_7 = section and ('Item 7' in section or 'item 7' in section.lower())
            is_item_11 = section and ('Item 11' in section or 'item 11' in section.lower())
            headers = table_data.get('headers', [])
            rows = table_data.get('rows', [])
            
            # Get surrounding context (previous text section)
            context = None
            if section:
                # Try to find related text
                for text_data in text_chunks:
                    if text_data.get('section') == section:
                        context = text_data['text'][:500]  # First 500 chars for context
                        break
            
            
            # Special handling for Item 5 fee tables: create separate chunks with semantic tags
            # PLACED AFTER Item 6 to follow the working pattern
            elif is_item_5 and rows and headers:
                logger.info(f"Processing Item 5 fee table with {len(rows)} rows")
                
                # CRITICAL FIX: Process each CELL with its corresponding COLUMN HEADER
                # This maps amounts to territories based on column headers, not cell text
                
                for row_idx, row in enumerate(rows):
                    # Skip header rows (they contain territory names)
                    row_cells = [str(cell).strip() for cell in row if cell]
                    
                    # Check if this is a header row (contains "territory" words in first cells)
                    if any('territory' in str(cell).lower() for cell in row[:2]):
                        logger.debug(f"Skipping header row {row_idx}")
                        continue
                    
                    # Process each CELL in the row, paired with its column HEADER
                    for col_idx, (header, cell_value) in enumerate(zip(headers, row)):
                        if not cell_value or not str(cell_value).strip():
                            continue  # Skip empty cells
                        
                        cell_value_str = str(cell_value).strip()
                        header_str = str(header).strip().lower()
                        
                        # IMPORTANT: Only process cells that contain numbers/amounts
                        # (Skip row headers like "Franchise Fee", "Cumulative", etc.)
                        if not any(c.isdigit() for c in cell_value_str):
                            continue
                        
                        # ===== DETERMINE SCOPE FROM COLUMN HEADER (NOT ROW TEXT) =====
                        scope_tag = None
                        if 'standard' in header_str and 'territory' in header_str:
                            scope_tag = '[SCOPE: Standard Territory]'
                        elif 'targeted' in header_str and 'territory' in header_str:
                            scope_tag = '[SCOPE: Targeted Territory]'
                        elif 'area development' in header_str or 'area develop' in header_str:
                            scope_tag = '[SCOPE: Area Development]'
                        
                        # ===== DETERMINE FEE TYPE FROM ROW CONTEXT =====
                        fee_type_tag = None
                        row_label = " ".join(row_cells).lower() if row_cells else ""
                        
                        if 'cumulative' in row_label or 'additional' in row_label:
                            if '2' in row_label or 'two' in row_label or 'second' in row_label:
                                fee_type_tag = '[FEE_TYPE: Cumulative Franchise Fee - 2 Territories]'
                            elif '3' in row_label or 'three' in row_label or 'third' in row_label:
                                fee_type_tag = '[FEE_TYPE: Cumulative Franchise Fee - 3 Territories]'
                            else:
                                fee_type_tag = '[FEE_TYPE: Cumulative Franchise Fee]'
                        elif 'initial' in row_label and 'fee' in row_label:
                            fee_type_tag = '[FEE_TYPE: Initial Franchise Fee]'
                        elif 'mailer' in row_label:
                            fee_type_tag = '[FEE_TYPE: Mailer Program Fee]'
                        elif 'digital' in row_label or 'seo' in row_label or ('marketing' in row_label and 'digital' in row_label):
                            fee_type_tag = '[FEE_TYPE: Digital Marketing Fee]'
                        elif 'technology' in row_label or 'tech' in row_label:
                            fee_type_tag = '[FEE_TYPE: Technology Fee]'
                        elif 'discount' in row_label or 'promotion' in row_label or 'heroes' in row_label or 'vetfran' in row_label:
                            fee_type_tag = '[FEE_TYPE: Discount/Promotion]'
                        else:
                            fee_type_tag = '[FEE_TYPE: Fee]'
                        
                        # ===== BUILD SEMANTIC TAGS =====
                        semantic_tags = []
                        semantic_tags.append('[SECTION: Item 5 - Initial Fees]')
                        if scope_tag:
                            semantic_tags.append(scope_tag)
                        if fee_type_tag:
                            semantic_tags.append(fee_type_tag)
                        semantic_tags.append('[FREQUENCY: One-Time]')  # Item 5 fees are one-time
                        
                        tags_str = " ".join(semantic_tags)
                        
                        # ===== CREATE ENRICHED CONTENT =====
                        enriched_content = f"{tags_str}\n{cell_value_str}\nColumn Header: {header_str}"
                        
                        # ===== CREATE CHUNK WITH METADATA =====
                        chunk_id = f"{document_name.lower().replace(' ', '_')}_item5_row{row_idx}_col{col_idx}"
                        
                        chunk_metadata = {
                            'column_index': col_idx,
                            'row_index': row_idx,
                            'item_number': '5',
                            'fee_type': fee_type_tag.replace('[FEE_TYPE: ', '').replace(']', '') if fee_type_tag else None,
                            'scope': scope_tag.replace('[SCOPE: ', '').replace(']', '') if scope_tag else None,
                            'fee_amount': cell_value_str,
                            'column_header': header_str,
                            'row_label': row_label
                        }
                        
                        chunk = DocumentChunk(
                            content=enriched_content,
                            chunk_id=chunk_id,
                            document_name=document_name,
                            chunk_type='table',
                            table_name='Item_5_Fees',
                            table_headers=headers,
                            row_count=len(rows),
                            section=section,
                            metadata=chunk_metadata
                        )
                        all_chunks.append(chunk)
                        logger.debug(f"Created Item 5 chunk: {scope_tag or 'Unknown Scope'} - {fee_type_tag} - {cell_value_str}")
                    
                    chunk_counter += 1
                
                continue  # Skip to next table
            
            # Special handling for Item 11 training tables
            elif is_item_11 and rows and headers:
                logger.info(f"Processing Item 11 training table with {len(rows)} rows")
                
                # Create chunks for training requirements
                for row_idx, row in enumerate(rows):
                    row_text = " ".join(str(cell) for cell in row if cell)
                    row_text_lower = row_text.lower()
                    
                    # Build semantic tags
                    semantic_tags = "[SECTION: Item 11 - Training & Support]"
                    
                    if any(term in row_text_lower for term in ['training', 'course', 'instruction', 'program']):
                        semantic_tags += " [TYPE: Training Requirement]"
                        # Try to extract duration
                        duration_match = re.search(r'(\d+\s+(?:day|week|month|hour))', row_text_lower)
                        if duration_match:
                            semantic_tags += f" [DURATION: {duration_match.group(1)}]"
                    elif any(term in row_text_lower for term in ['assistance', 'support', 'help']):
                        semantic_tags += " [TYPE: Support]"
                        if 'ongoing' in row_text_lower:
                            semantic_tags += " [CATEGORY: Ongoing]"
                        elif 'initial' in row_text_lower:
                            semantic_tags += " [CATEGORY: Initial]"
                    
                    # Format row
                    row_content = " | ".join(str(cell) for cell in row if cell)
                    enriched_content = f"{semantic_tags}\nSection: {section}\n{row_content}"
                    
                    chunk_id = f"{document_name}_table_{chunk_counter}_row_{row_idx}"
                    
                    chunk = DocumentChunk(
                        content=enriched_content,
                        chunk_id=chunk_id,
                        document_name=document_name,
                        chunk_type='table',
                        table_name=table_data.get('table_name'),
                        table_headers=headers,
                        row_count=1,
                        section=section,
                        metadata={
                            'column_count': table_data.get('column_count'),
                            'table_index': len([c for c in all_chunks if c.chunk_type == 'table']),
                            'row_index': row_idx,
                            'item_number': '11'
                        }
                    )
                    all_chunks.append(chunk)
                    chunk_counter += 1
                
                continue  # Skip to next table
            
            # Special handling for Item 7 investment tables: keep as SINGLE complete chunk
            # This preserves number ordering and prevents corruption
            if is_item_7 and rows and headers:
                logger.info(f"Processing Item 7 investment table with {len(rows)} rows - keeping as single complete chunk")
                
                # Format entire table as one chunk to preserve ordering
                table_content = self._format_table_for_chunk(table_data, context)
                
                chunk_id = f"{document_name}_table_{chunk_counter}"
                chunk = DocumentChunk(
                    content=table_content,
                    chunk_id=chunk_id,
                    document_name=document_name,
                    chunk_type='table',
                    table_name=table_data.get('table_name'),
                    table_headers=headers,
                    row_count=len(rows),
                    section=section,
                    metadata={
                        'column_count': table_data.get('column_count'),
                        'table_index': len([c for c in all_chunks if c.chunk_type == 'table']),
                        'is_investment_table': True,  # Mark as investment table
                        'preserve_ordering': True     # Flag to preserve number order
                    }
                )
                all_chunks.append(chunk)
                chunk_counter += 1
                continue  # Skip to next table
            
            # Special handling for Item 6 fee tables: create separate chunks for each fee row
            elif is_item_6 and rows and headers:
                logger.info(f"Processing Item 6 fee table with {len(rows)} rows")
                
                # Create a separate chunk for each fee row
                for row_idx, row in enumerate(rows):
                    # Check if this row is a discount/promotion vs base fee
                    row_text = " ".join(str(cell) for cell in row if cell).lower()
                    is_discount = any(term in row_text for term in ['discount', 'promotion', 'promo', 'community heroes', 'vetfran', 'may apply', 'qualifying'])
                    is_base_fee = not is_discount and any(term in row_text for term in ['initial franchise fee', 'royalty', 'marketing', 'fee', 'amount'])
                    
                    # Extract fee information from this row
                    fee_info = self._extract_fee_info_from_row(row, headers)
                    
                    # Format this row as a searchable chunk
                    row_content = self._format_fee_row_chunk(row, headers, section, fee_info)
                    
                    # Semantic tags are already added in _format_fee_row_chunk
                    # But ensure base fees and discounts are clearly separated
                    # The [TYPE: Base Fee] vs [TYPE: Discount] tags handle this
                    
                    # Create chunk ID
                    chunk_id = f"{document_name}_table_{chunk_counter}_row_{row_idx}"
                    
                    # Extract item number from section (e.g., "Item 6" -> "6")
                    item_number = None
                    if section:
                        item_match = re.search(r'Item\s+(\d+)', section, re.IGNORECASE)
                        if item_match:
                            item_number = item_match.group(1)
                    
                    # Build metadata with fee information
                    chunk_metadata = {
                        'column_count': table_data.get('column_count'),
                        'table_index': len([c for c in all_chunks if c.chunk_type == 'table']),
                        'row_index': row_idx,
                        'item_number': item_number
                    }
                    
                    # Add fee-specific metadata if available
                    if fee_info.get('fee_name'):
                        chunk_metadata['fee_name'] = fee_info['fee_name']
                    if fee_info.get('fee_amount'):
                        chunk_metadata['fee_amount'] = fee_info['fee_amount']
                    
                    chunk = DocumentChunk(
                        content=row_content,
                        chunk_id=chunk_id,
                        document_name=document_name,
                        chunk_type='table',
                        table_name=table_data.get('table_name'),
                        table_headers=headers,
                        row_count=1,  # Each row is its own chunk
                        section=section,
                        metadata=chunk_metadata
                    )
                    all_chunks.append(chunk)
                    chunk_counter += 1
                    
                    logger.debug(f"Created fee chunk: {fee_info.get('fee_name', 'Unknown')} = {fee_info.get('fee_amount', 'N/A')}")
            
            else:
                # For non-Item-6 tables, or if table structure is unclear, use original chunking
                # But still try to create row-based chunks if it's a fee-related table
                is_fee_table = False
                if section:
                    section_lower = section.lower()
                    is_fee_table = any(keyword in section_lower for keyword in ['fee', 'cost', 'payment', 'charge', 'royalty'])
                
                # If it's a fee table but not Item 6, still create row-based chunks
                if is_fee_table and rows and headers:
                    logger.info(f"Processing fee table (non-Item 6) with {len(rows)} rows")
                    
                    for row_idx, row in enumerate(rows):
                        fee_info = self._extract_fee_info_from_row(row, headers)
                        row_content = self._format_fee_row_chunk(row, headers, section, fee_info)
                        
                        chunk_id = f"{document_name}_table_{chunk_counter}_row_{row_idx}"
                        
                        chunk_metadata = {
                            'column_count': table_data.get('column_count'),
                            'table_index': len([c for c in all_chunks if c.chunk_type == 'table']),
                            'row_index': row_idx
                        }
                        
                        if fee_info.get('fee_name'):
                            chunk_metadata['fee_name'] = fee_info['fee_name']
                        if fee_info.get('fee_amount'):
                            chunk_metadata['fee_amount'] = fee_info['fee_amount']
                        
                        chunk = DocumentChunk(
                            content=row_content,
                            chunk_id=chunk_id,
                            document_name=document_name,
                            chunk_type='table',
                            table_name=table_data.get('table_name'),
                            table_headers=headers,
                            row_count=1,
                            section=section,
                            metadata=chunk_metadata
                        )
                        all_chunks.append(chunk)
                        chunk_counter += 1
                else:
                    # Original behavior: single chunk for entire table
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
    
    def _is_section_header(self, line: str) -> bool:
        """
        Check if a line is a section header (e.g., "ITEM 1", "ITEM 2", etc.).
        
        Args:
            line: Line of text to check
            
        Returns:
            True if line appears to be a section header
        """
        line_upper = line.strip().upper()
        # Match patterns like "ITEM 1", "ITEM 1:", "ITEM 1.", "ITEM 1 -", etc.
        if re.match(r'^ITEM\s+\d+', line_upper):
            return True
        # Also match numbered sections like "1.", "2.", "3.", etc. at start of line
        if re.match(r'^\d+\.\s+[A-Z]', line_upper):
            return True
        return False
    
    def _is_list_item(self, line: str) -> bool:
        """
        Check if a line is part of a numbered or bulleted list.
        
        Args:
            line: Line of text to check
            
        Returns:
            True if line appears to be a list item
        """
        line_stripped = line.strip()
        # Match numbered lists: "1.", "2.", "Year 1:", "Year 2:", etc.
        if re.match(r'^(\d+\.|Year\s+\d+:|•|\-|\*)\s+', line_stripped, re.IGNORECASE):
            return True
        # Match patterns like "Year 1", "Year 2", "Year 3", etc.
        if re.match(r'^Year\s+\d+', line_stripped, re.IGNORECASE):
            return True
        return False
    
    def _is_table_row(self, line: str) -> bool:
        """
        Check if a line appears to be a table row.
        
        Args:
            line: Line of text to check
            
        Returns:
            True if line appears to be a table row
        """
        line_stripped = line.strip()
        if not line_stripped:
            return False
        
        # Table rows typically have:
        # - Multiple pipe characters (|)
        # - Multiple tab characters (\t)
        # - Multiple dollar signs or numbers in a row pattern
        pipe_count = line_stripped.count('|')
        tab_count = line_stripped.count('\t')
        
        # If it has 2+ pipes or 2+ tabs, likely a table row
        if pipe_count >= 2 or tab_count >= 2:
            return True
        
        # Check for pattern like "$X | $Y | $Z" or "X\tY\tZ" (tab-separated)
        if tab_count >= 1 and len(line_stripped.split('\t')) >= 3:
            return True
        
        return False
    
    def _is_incomplete_chunk(self, chunk: str) -> bool:
        """
        Detect if a chunk appears to be incomplete (cut off mid-sentence, mid-list, etc.).
        
        Args:
            chunk: Chunk text to check
            
        Returns:
            True if chunk appears incomplete
        """
        if not chunk or len(chunk.strip()) < 10:
            return False
        
        chunk_stripped = chunk.strip()
        
        # Check for cut-off patterns
        # 1. Starts with "...or $", "...and", etc. (mid-sentence)
        if re.match(r'^\.\.\.(or|and|\$)', chunk_stripped, re.IGNORECASE):
            return True
        
        # 2. Ends with incomplete sentence (no punctuation, but has content)
        # Exception: if it ends with a colon, it might be a header
        if not chunk_stripped[-1] in '.!?;:' and len(chunk_stripped) > 50:
            # Check if it's likely incomplete (has words but no ending punctuation)
            words = chunk_stripped.split()
            if len(words) > 5:  # Has substantial content
                return True
        
        # 3. Contains incomplete list (e.g., "Year 1, Year 2, Year" - missing Year 3)
        # This is harder to detect, but we can check for patterns like "Year 1, Year 2" without Year 3
        year_pattern = r'Year\s+(\d+)'
        years_found = re.findall(year_pattern, chunk_stripped, re.IGNORECASE)
        if years_found:
            years = [int(y) for y in years_found if y.isdigit()]
            if years:
                min_year = min(years)
                max_year = max(years)
                # If we have Year 1 and Year 4 but not Year 2 or 3, it's incomplete
                expected_years = set(range(min_year, max_year + 1))
                found_years = set(years)
                if expected_years != found_years:
                    return True
        
        # 4. Ends with incomplete table row indicators
        if chunk_stripped.endswith('|') or chunk_stripped.endswith('| '):
            return True
        
        return False
    
    def _split_text(self, text: str, target_tokens: int) -> List[str]:
        """
        Split text into chunks. SIMPLE APPROACH:
        - Split ONLY at ITEM headers
        - Keep complete tables together (even if 2000+ chars)
        - Keep complete lists together
        - Never split mid-row or mid-item
        
        Args:
            text: Text to split
            target_tokens: Target number of tokens per chunk (used as guide, not strict limit)
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Count tokens
        tokens = self._count_tokens(text)
        
        # If text fits in one chunk, return as is
        if tokens <= target_tokens * 1.5:  # Allow 50% overflow to keep things together
            return [text]
        
        # Split by ITEM headers first (most important boundary)
        item_pattern = r'(ITEM\s+\d+[:\s])'
        parts = re.split(item_pattern, text, flags=re.IGNORECASE)
        
        chunks = []
        current_chunk = ""
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Check if this part is an ITEM header
            if re.match(item_pattern, part, re.IGNORECASE):
                # Save previous chunk if it exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Start new chunk with ITEM header
                current_chunk = part
                # Add content after header if exists
                if i + 1 < len(parts):
                    current_chunk += parts[i + 1]
                    i += 2
                else:
                    i += 1
                continue
            
            # Regular content - add to current chunk
            current_chunk += part
            i += 1
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Post-process: For chunks that are still too large, split at paragraph breaks
        # BUT never split tables or lists
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            
            # If chunk is reasonable size, keep it
            if chunk_tokens <= target_tokens * 2:  # Allow 2x overflow for complete tables
                final_chunks.append(chunk)
                continue
            
            # Chunk is very large - try to split at paragraph breaks only
            # But preserve tables and lists
            lines = chunk.split('\n')
            current_subchunk = []
            current_subchunk_tokens = 0
            in_table = False
            in_list = False
            
            for line in lines:
                line_tokens = self._count_tokens(line)
                is_table_line = self._is_table_row(line)
                is_list_line = self._is_list_item(line)
                
                # Track table/list context
                if is_table_line:
                    in_table = True
                elif not is_table_line and in_table and not line.strip():
                    # Empty line after table - table ended
                    in_table = False
                
                if is_list_line:
                    in_list = True
                elif not is_list_line and in_list and not line.strip():
                    # Empty line after list - list ended
                    in_list = False
                
                # If we're in a table or list, keep adding even if it exceeds limit
                if in_table or in_list:
                    current_subchunk.append(line)
                    current_subchunk_tokens += line_tokens
                elif current_subchunk_tokens + line_tokens > target_tokens * 1.5 and current_subchunk:
                    # Save current subchunk and start new one
                    final_chunks.append('\n'.join(current_subchunk))
                    current_subchunk = [line]
                    current_subchunk_tokens = line_tokens
                else:
                    current_subchunk.append(line)
                    current_subchunk_tokens += line_tokens
            
            # Add remaining subchunk
            if current_subchunk:
                final_chunks.append('\n'.join(current_subchunk))
        
        return final_chunks if final_chunks else [text]
    
    def _batch_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Generate embeddings in batches to optimize API calls.
        
        THIS IS WHERE VECTORIZATION HAPPENS!
        
        Instead of calling OpenAI API once per text (slow and expensive),
        we process multiple texts at once in batches.
        
        Example:
        Input:  ["chunk 1", "chunk 2", ..., "chunk 1000"]
        Output: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]  (1000 embeddings)
        
        Args:
            texts: List of texts to embed (document chunks)
            batch_size: Number of texts per batch (50 is optimal for OpenAI API)
            
        Returns:
            List of embedding vectors (each is a list of 1536 numbers)
        """
        all_embeddings = []
        
        # Process texts in batches (e.g., 50 at a time)
        # This is much faster than one-by-one
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]  # Get next batch of texts
            
            try:
                # Call OpenAI API to convert batch of texts to embeddings
                # This is the actual vectorization: text → numbers
                response = self.client.embeddings.create(
                    model=self.embedding_model,  # "text-embedding-3-small"
                    input=batch                  # List of texts (e.g., 50 texts)
                )
                # Extract embeddings from response
                # Each text becomes a list of 1536 numbers
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)  # Add to our collection
                logger.info(f"Generated embeddings for batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
                
            except Exception as e:
                # If batch fails, try individual texts (slower but more reliable)
                logger.error(f"Error generating embeddings for batch: {e}")
                # Retry with individual calls
                for text in batch:
                    try:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=[text]  # Just one text
                        )
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as retry_error:
                        logger.error(f"Error embedding individual text: {retry_error}")
                        # Use zero vector as fallback (not ideal, but better than crashing)
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
        
        IMPORTANT: Always loads with embedding function to ensure queries work correctly.
        Without the embedding function, ChromaDB might use a default that doesn't match
        the stored embeddings, causing dimension mismatch errors.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Always load with embedding function to ensure compatibility
            # This ensures queries will use the same embedding model (1536 dimensions)
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function  # Always specify!
                )
            except Exception as e:
                # If loading with embedding function fails, try without (for diagnostics)
                logger.warning(f"Could not load collection with embedding function: {e}")
                try:
                    self.collection = self.chroma_client.get_collection(
                        name=self.collection_name
                    )
                    logger.warning("Loaded collection without embedding function - queries may fail!")
                    logger.warning("If you get dimension errors, rebuild: python rebuild_vector_store.py")
                except Exception as e2:
                    logger.error(f"Failed to load collection: {e2}")
                    return False
            
            count = self.collection.count()
            logger.info(f"Loaded vector store with {count} chunks")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def _expand_query(self, question: str) -> List[str]:
        """
        Expand query with comprehensive synonyms and variations to improve retrieval.
        This is UNIVERSAL - works for all 26 franchise documents.
        
        Examples:
        - "Standard Territory" → also search "25000 children", "25,000 children", "25000+ children"
        - "Year 2 royalty" → also search "second full calendar", "months 13-24", "year two"
        - "Initial Franchise Fee" → also search "franchise fee", "initial fee", "upfront fee"
        - "litigation" → also search "Item 3", "ITEM 3 LITIGATION", "disclosure requirements"
        
        Args:
            question: Original user question
            
        Returns:
            List of query expansion terms
        """
        expansions = []
        question_lower = question.lower()
        
        # ===== TERRITORY EXPANSIONS =====
        # Only add territory number expansions for TERRITORY DEFINITION questions, not fee questions
        # This prevents polluting fee searches with investment numbers
        is_territory_definition = any(term in question_lower for term in ['territory', 'defined', 'definition', 'size', 'population', 'children', 'what is a standard territory', 'what is a targeted territory'])
        is_fee_question = any(term in question_lower for term in ['fee', 'cost', 'price', 'amount', 'investment', 'royalty', 'marketing', 'initial franchise'])
        
        # Only expand with numbers if it's a territory definition question, NOT a fee question
        if is_territory_definition and not is_fee_question:
            if any(term in question_lower for term in ['standard territory', 'standard']):
                expansions.extend([
                    '25000 children', '25,000 children', '25000+ children', '25k children',
                    '25000 children under 10', 'up to 25000 children', 'standard territory definition',
                    'standard territory size', 'standard territory population'
                ])
            
            if any(term in question_lower for term in ['targeted territory', 'targeted']):
                expansions.extend([
                    'targeted territory definition', 'targeted territory size',
                    'targeted territory population', 'smaller territory'
                ])
        elif is_territory_definition and is_fee_question:
            # For fee questions that mention territory, add territory keywords but NOT numbers
            if any(term in question_lower for term in ['standard territory', 'standard']):
                expansions.extend([
                    'standard territory', 'standard', 'standard territory fee'
                ])
            
            if any(term in question_lower for term in ['targeted territory', 'targeted']):
                expansions.extend([
                    'targeted territory', 'targeted', 'targeted territory fee'
                ])
        
        # ===== YEAR/PERIOD EXPANSIONS (COMPREHENSIVE) =====
        year_pattern = re.search(r'year\s+(\d+)', question_lower)
        if year_pattern:
            year_num = int(year_pattern.group(1))
            # Map year number to ALL possible variations
            year_expansions = {
                1: [
                    'first full calendar', 'first full calendar year', 'first year',
                    'year one', 'year 1', 'months 1-12', 'months 1 to 12',
                    'first calendar year', 'initial year', 'year 1 period'
                ],
                2: [
                    'second full calendar', 'second full calendar year', 'second year',
                    'year two', 'year 2', 'months 13-24', 'months 13 to 24',
                    'second calendar year', 'year 2 period', 'months thirteen to twenty four'
                ],
                3: [
                    'third full calendar', 'third full calendar year', 'third year',
                    'year three', 'year 3', 'months 25-36', 'months 25 to 36',
                    'third calendar year', 'year 3 period', 'months twenty five to thirty six'
                ],
                4: [
                    'fourth full calendar', 'fourth full calendar year', 'fourth year',
                    'year four', 'year 4', 'months 37-48', 'months 37 to 48',
                    'fourth calendar year', 'year 4 period', 'months thirty seven to forty eight'
                ],
                5: [
                    'fifth full calendar', 'fifth full calendar year', 'fifth year',
                    'year five', 'year 5', 'months 49+', 'months 49 and beyond',
                    'fifth calendar year', 'year 5 period', 'year 5 and beyond'
                ]
            }
            if year_num in year_expansions:
                expansions.extend(year_expansions[year_num])
        
        # Also detect ordinal patterns (first, second, third, etc.)
        ordinal_pattern = re.search(r'\b(first|second|third|fourth|fifth)\s+(full\s+)?calendar\s+year', question_lower)
        if ordinal_pattern:
            ordinal_map = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5}
            year_num = ordinal_map.get(ordinal_pattern.group(1))
            if year_num:
                year_expansions = {
                    1: ['year 1', 'year one', 'months 1-12'],
                    2: ['year 2', 'year two', 'months 13-24'],
                    3: ['year 3', 'year three', 'months 25-36'],
                    4: ['year 4', 'year four', 'months 37-48'],
                    5: ['year 5', 'year five', 'months 49+']
                }
                if year_num in year_expansions:
                    expansions.extend(year_expansions[year_num])
        
        # Month range patterns
        month_pattern = re.search(r'months?\s+(\d+)[-\s]+(\d+)', question_lower)
        if month_pattern:
            start_month = int(month_pattern.group(1))
            if start_month <= 12:
                expansions.extend(['year 1', 'first full calendar year', 'first year'])
            elif start_month <= 24:
                expansions.extend(['year 2', 'second full calendar year', 'second year'])
            elif start_month <= 36:
                expansions.extend(['year 3', 'third full calendar year', 'third year'])
            elif start_month <= 48:
                expansions.extend(['year 4', 'fourth full calendar year', 'fourth year'])
        
        # ===== FEE TYPE EXPANSIONS =====
        if 'royalty' in question_lower:
            expansions.extend([
                'continuing royalty', 'royalty fee', 'royalty rate', 'royalty percentage',
                'royalty amount', 'royalty payment', 'ongoing royalty', 'royalty structure'
            ])
        
        if 'franchise fee' in question_lower or 'initial fee' in question_lower:
            expansions.extend([
                'franchise fee', 'initial franchise fee', 'initial fee', 'upfront fee',
                'franchise fee amount', 'initial payment', 'franchise fee payment'
            ])
        
        if 'marketing fee' in question_lower or 'marketing fund' in question_lower:
            expansions.extend([
                'marketing fund', 'marketing contribution', 'advertising fee',
                'marketing fee', 'marketing fund contribution', 'advertising contribution'
            ])
        
        if 'chatbot fee' in question_lower or 'chatbot' in question_lower:
            expansions.extend([
                'chatbot fee', 'chatbot', 'chat bot fee', 'chat bot',
                'chatbot service fee', 'chatbot monthly fee'
            ])
        
        if 'technology fee' in question_lower or 'tech fee' in question_lower:
            expansions.extend([
                'technology fee', 'tech fee', 'technology', 'tech',
                'technology service fee', 'tech service fee'
            ])
        
        # ===== PERIOD/DURATION EXPANSIONS =====
        if any(term in question_lower for term in ['monthly', 'per month', 'month']):
            expansions.extend([
                'monthly', 'each month', 'per month', 'monthly fee',
                'monthly payment', 'monthly charge', 'monthly cost'
            ])
        
        if any(term in question_lower for term in ['annual', 'yearly', 'per year']):
            expansions.extend([
                'annual', 'yearly', 'per year', 'annual fee',
                'annual payment', 'annual charge', 'yearly fee'
            ])
        
        if any(term in question_lower for term in ['one-time', 'one time', 'initial', 'setup']):
            expansions.extend([
                'one-time', 'one time', 'initial', 'setup',
                'one-time fee', 'one time payment', 'setup fee', 'initial payment'
            ])
        
        # ===== INVESTMENT/RANGE EXPANSIONS =====
        if any(term in question_lower for term in ['investment', 'initial investment', 'cost', 'startup']):
            expansions.extend([
                'initial investment', 'startup cost', 'total investment', 'capital requirement',
                'investment amount', 'investment range', 'startup investment', 'total cost'
            ])
        
        if 'range' in question_lower or ('to' in question_lower and '$' in question_lower):
            expansions.extend([
                'range', 'minimum to maximum', 'low to high',
                'investment range', 'cost range', 'fee range', 'amount range'
            ])
        
        # ===== ITEM-SPECIFIC EXPANSIONS =====
        # Litigation/Item 3 expansions
        if any(term in question_lower for term in ['litigation', 'lawsuit', 'legal action', 'court case', 'sued', 'suing']):
            expansions.extend([
                'Item 3', 'ITEM 3', 'Item 3 Litigation', 'ITEM 3 LITIGATION',
                'disclosure requirements', 'litigation disclosure', 'legal proceedings',
                'Item 3 disclosure', 'litigation history', 'legal actions'
            ])
        
        # Bankruptcy/Item 4 expansions
        if any(term in question_lower for term in ['bankruptcy', 'bankrupt', 'chapter 7', 'chapter 13', 'chapter 11']):
            expansions.extend([
                'Item 4', 'ITEM 4', 'Item 4 Bankruptcy', 'ITEM 4 BANKRUPTCY',
                'bankruptcy disclosure', 'bankruptcy filing', 'bankruptcy history',
                'Item 4 disclosure', 'chapter 7 bankruptcy', 'chapter 13 bankruptcy'
            ])
        
        # Financial Performance/Item 19 expansions
        if any(term in question_lower for term in ['financial performance', 'earnings', 'revenue', 'sales', 'income']):
            expansions.extend([
                'Item 19', 'ITEM 19', 'Item 19 Financial Performance',
                'financial performance representations', 'earnings claims', 'revenue data'
            ])
        
        # Item number expansions (if user mentions specific item)
        item_match = re.search(r'item\s+(\d+)', question_lower)
        if item_match:
            item_num = item_match.group(1)
            expansions.extend([
                f'Item {item_num}', f'ITEM {item_num}',
                f'Item {item_num} disclosure', f'ITEM {item_num} disclosure'
            ])
        
        return expansions
    
    def query(
        self,
        question: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a RAG query: retrieve relevant context and generate response.
        
        THIS IS THE MAIN RAG PIPELINE METHOD!
        
        Complete RAG Flow:
        1. Check for exact Q&A match (fast path)
        2. Convert question to embedding (text → numbers)
        3. Vector search: Find similar document chunks
        4. Build context from retrieved chunks
        5. Send to AI: Generate answer based on context
        6. Return answer with sources
        
        Args:
            question: User's question (e.g., "What franchises are available?")
            include_metadata: Whether to include metadata in response
            
        Returns:
            Dictionary with:
            - answer: The AI-generated answer
            - sources: Which document chunks were used
            - metadata: Additional info (model, retrieval count, etc.)
        """
        try:
            # ===== STEP 1: FAST PATH - CHECK Q&A MATCHER =====
            # Some questions have pre-written exact answers (faster and more accurate)
            # This bypasses vector search and AI generation for known questions
            if self.qa_matcher:
                exact_answer = self.qa_matcher.get_exact_answer(question)
                if exact_answer:
                    logger.info(f"Returning exact answer from Q&A knowledge base")
                    return {
                        'answer': exact_answer,
                        'sources': [{
                            'content': 'Q&A Knowledge Base',
                            'metadata': {'source_type': 'exact_qa_match'},
                            'relevance_score': 1.0
                        }],
                        'metadata': {
                            'model': 'exact_match',
                            'retrieval_count': 0,
                            'temperature': 0,
                            'source': 'qa_knowledge_base'
                        }
                    }
            # ===== VECTOR SEARCH - FIND SIMILAR CHUNKS =====
            # This is where the magic happens! ChromaDB:
            # 1. Converts your question to an embedding (using embedding_function)
            # 2. Compares it with ALL stored document chunk embeddings
            # 3. Finds the chunks with most similar embeddings (smallest distance)
            # 4. Returns the top N most similar chunks
            #
            # How similarity works:
            # - Similar meanings = similar embeddings = small distance
            # - Different meanings = different embeddings = large distance
            # - ChromaDB calculates "distance" between embeddings mathematically
            #
            # Example:
            # Question: "What franchises are available?"
            # → Converts to embedding: [0.123, -0.456, ...]
            # → Compares with all stored embeddings
            # → Finds chunks about "franchise opportunities", "available franchises", etc.
            # → Returns top 15 most similar chunks
            
            # ===== QUERY EXPANSION =====
            # Expand query with synonyms and variations to improve retrieval
            expanded_queries = self._expand_query(question)
            
            # ===== SIMPLE RETRIEVAL =====
            # Detect Item number if mentioned, retrieve 20-25 chunks, let similarity work naturally
            question_lower = question.lower()
            mentioned_items = []
            
            # Simple regex to detect "Item X" or "Item X:"
            item_match = re.search(r'item\s+(\d+)', question_lower)
            if item_match:
                item_num = int(item_match.group(1))
                mentioned_items.append(item_num)
            
            # Retrieve 20-25 chunks (simple, no fancy logic)
            retrieval_n = 25  # Fixed at 25 chunks for all queries
            if mentioned_items:
                logger.info(f"Query mentions Item {mentioned_items[0]}, retrieving {retrieval_n} chunks")
            
            # Use expanded query for better retrieval
            # Combine original query with expansions for richer embedding
            query_for_embedding = question
            if expanded_queries:
                # Combine original + expansions for better semantic matching
                query_for_embedding = question + " " + " ".join(expanded_queries)
                logger.info(f"Query expanded with: {expanded_queries[:3]}...")  # Log first 3
            
            # Manually create embedding to avoid ChromaDB's embed_query issues
            # Convert question to embedding using our embedding function
            question_embedding = self.embedding_function.__call__([query_for_embedding])
            
            # Use query_embeddings instead of query_texts to avoid embed_query method issues
            results = self.collection.query(
                query_embeddings=question_embedding,  # Pre-computed embedding (list of lists)
                n_results=retrieval_n                 # Get top N most similar chunks
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
            
            # ===== DETAILED LOGGING FOR DEBUGGING =====
            logger.info("\n" + "=" * 80)
            logger.info(f"=== QUERY: {question} ===")
            logger.info("=" * 80)
            
            for i, chunk in enumerate(retrieved_chunks):
                metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
                distance = retrieved_distances[i] if i < len(retrieved_distances) else 1.0
                
                chunk_id = metadata.get('chunk_id', 'Unknown')
                document_name = metadata.get('document_name', 'Unknown')
                section = metadata.get('section', 'N/A')
                chunk_text_preview = chunk[:400] if chunk else "Empty chunk"
                
                logger.info(f"\n=== CHUNK {i+1} ===")
                logger.info(f"ID: {chunk_id}")
                logger.info(f"Document: {document_name}")
                logger.info(f"Section: {section}")
                logger.info(f"Distance: {distance:.6f}")
                logger.info(f"Text: {chunk_text_preview}...")
                logger.info("-" * 80)
            
            logger.info("=" * 80 + "\n")
            # ===== END DETAILED LOGGING =====
            
            # Separate Q&A documents from FDD documents and prioritize Q&A
            # Do NOT boost Item chunks - let similarity ranking work naturally
            qa_chunks = []
            fdd_chunks = []
            
            for i, chunk in enumerate(retrieved_chunks):
                metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
                document_name = metadata.get('document_name', 'Unknown')
                section = metadata.get('section', '')
                distance = retrieved_distances[i] if i < len(retrieved_distances) else 1.0
                
                chunk_info = {
                    'chunk': chunk,
                    'metadata': metadata,
                    'distance': distance,
                    'index': i
                }
                
                # Check if it's a Q&A document
                if any(keyword in document_name for keyword in ['Knowledge', 'AI Bot', 'training', 'Question']):
                    qa_chunks.append(chunk_info)
                else:
                    fdd_chunks.append(chunk_info)
            
            # Prioritize: Q&A first, then FDD chunks (in similarity order, no boosting)
            prioritized_chunks = qa_chunks + fdd_chunks
            
            # Log if Item chunks were retrieved (for debugging)
            if mentioned_items:
                item_chunks_found = []
                for chunk_info in prioritized_chunks:
                    section = chunk_info['metadata'].get('section', '')
                    if section:
                        section_upper = section.upper()
                        for item_num in mentioned_items:
                            if f'ITEM {item_num}' in section_upper or f'ITEM {item_num}:' in section_upper:
                                item_chunks_found.append(item_num)
                                break
                if item_chunks_found:
                    logger.info(f"Retrieved chunks for Item(s) {list(set(item_chunks_found))} (no boosting applied)")
            
            context_parts = []
            for i, chunk_info in enumerate(prioritized_chunks):
                chunk = chunk_info['chunk']
                metadata = chunk_info['metadata']
                chunk_type = metadata.get('chunk_type', 'text')
                table_name = metadata.get('table_name')
                section = metadata.get('section')
                document_name = metadata.get('document_name', 'Unknown')
                
                # Highlight Q&A documents
                is_qa = any(keyword in document_name for keyword in ['Knowledge', 'AI Bot', 'training', 'Question'])
                qa_marker = " [Q&A DOCUMENT - USE EXACT TEXT]" if is_qa else ""
                
                context_header = f"\n--- Source {i+1}{qa_marker} [Document: {document_name}] "
                if chunk_type == 'table' and table_name:
                    context_header += f"(Table: {table_name}) "
                if section:
                    context_header += f"[Section: {section}] "
                context_header += "---\n"
                
                context_parts.append(context_header + chunk)
            
            context = "\n".join(context_parts)
            
            # ===== STEP 6: PREPARE AI PROMPT =====
            # Extract document names from metadata for better context
            # This helps the AI understand which documents it's reading from
            document_names = set()
            for metadata in retrieved_metadatas:
                doc_name = metadata.get('document_name', '')
                if doc_name:
                    document_names.add(doc_name)
            
            # Build additional context about which documents are available
            document_context = ""
            if document_names:
                doc_list = ", ".join(sorted(document_names))
                document_context = f"\n\nAvailable documents in context: {doc_list}\n"
                # If question mentions a specific franchise, emphasize using only that document
                # This helps AI focus on the right document
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
            
            # Check if we have Q&A documents in the context
            has_qa_docs = any('[Q&A DOCUMENT - USE EXACT TEXT]' in part for part in context_parts)
            
            if has_qa_docs:
                # Special handling for Q&A documents - force exact answer extraction
                user_prompt = f"""🚨 CRITICAL INSTRUCTION - Q&A DOCUMENT DETECTED 🚨

The context below contains Q&A documents with pre-approved answers.

YOUR TASK:
1. Look for a question in the context that matches: "{question}"
2. Find the text that appears after "Answer:" for that question
3. Copy that answer text EXACTLY - word-for-word
4. Return ONLY that exact answer text
5. Do NOT add, modify, paraphrase, or combine with other sources

Context:
{context}

USER QUESTION: {question}

Extract and return the EXACT answer from the Q&A document above."""
            else:
                # Standard FDD document handling
                user_prompt = f"""Based on the following context from Franchise Disclosure Documents (FDDs), answer the question by READING what the documents say.

CORE RULE: READ, NOT INTERPRET

- If the FDD says something explicitly, report it exactly
- If the FDD doesn't say something, say: "The FDD does not explicitly state this."
- Do NOT guess, infer, synthesize, or use your training data to fill gaps
- Do NOT do math with numbers (no subtracting discounts)
- Do NOT interpret ambiguous text
- Just READ and REPORT

CRITICAL: COMPLETE INFORMATION REQUIREMENTS

Return COMPLETE tables and sections, never partial data.

- If the answer involves multiple rows (Year 1, 2, 3, 4), return ALL rows in order
- If the answer is a fee table, return the ENTIRE table with all rows
- If the answer is an investment table, return the ENTIRE table with all columns and rows
- NEVER return a partial table row - if a row appears incomplete, do not use it
- If table data appears corrupted or mixed (e.g., numbers in wrong order), do NOT use it - report that the data appears incomplete
- For fee structures: Include ALL years/periods (Year 1, Year 2, Year 3, Year 4, etc.) - if any year is missing, note that the data appears incomplete
- For investment ranges: Report the COMPLETE range exactly as shown (e.g., "$122,070 to $168,420") not partial or corrupted amounts

INSTRUCTIONS:
- Look through the context for information that directly answers the question
- Report what the FDD states, using exact wording when possible
- Cite the document name and section/table when reporting information
- If multiple documents mention the same thing, report what each says
- If information is not found, state: "The FDD does not explicitly state this."
- Do NOT combine or synthesize information to create new facts
- Do NOT infer meaning from ambiguous statements
- For tables and lists, ensure you report ALL items, not just a subset

{document_context}
Context from documents:
{context}

Question: {question}

Answer by reading and reporting what the FDD explicitly states. Ensure you return COMPLETE information for tables, lists, and fee structures. If the FDD does not state something, say so."""
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # ===== STEP 7: AI GENERATION - CREATE ANSWER =====
            # This is where the AI reads the context and generates an answer
            # 
            # What happens:
            # 1. AI receives: system prompt + context (document chunks) + user question
            # 2. AI reads and understands the context
            # 3. AI generates an answer based on the context (not from memory!)
            # 4. AI returns the answer
            #
            # This is the "Generation" part of RAG:
            # - Retrieval: We found relevant chunks (done above)
            # - Augmented: We added context to the question (done above)
            # - Generation: AI creates answer (happening here!)
            response = self.client.chat.completions.create(
                model=self.llm_model,              # "gpt-4o-mini" - fast and cheap
                messages=messages,                  # System prompt + context + question
                temperature=self.temperature,       # 0.0 = deterministic (exact answers)
                max_tokens=self.max_response_tokens  # Maximum answer length
            )
            
            # Extract the generated answer from the AI response
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

