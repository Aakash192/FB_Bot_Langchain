"""
Q&A Matcher - Returns exact answers from Q&A knowledge base
Similar to the JavaScript FAQ system
"""

import chromadb
from chromadb.config import Settings
import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    if not text:
        return ''
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text.lower())).strip()


class QAMatcher:
    """Matches user questions to Q&A knowledge base for exact answers."""
    
    def __init__(self, chroma_db_path: str = "chroma_db"):
        """Initialize the Q&A matcher."""
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = None
        self.qa_cache = []
        self._load_qa_documents()
    
    def _load_qa_documents(self):
        """Load all Q&A documents into memory for fast matching."""
        try:
            self.collection = self.chroma_client.get_collection(name="docx_rag_collection")
            
            # Get all items
            all_items = self.collection.get()
            
            # Filter for Q&A documents
            for i, metadata in enumerate(all_items['metadatas']):
                doc_name = metadata.get('document_name', '')
                if any(keyword in doc_name for keyword in ['Knowledge', 'AI Bot', 'training', 'Question']):
                    content = all_items['documents'][i]
                    # Parse Q&A pairs from content
                    qa_pairs = self._extract_qa_pairs(content)
                    self.qa_cache.extend(qa_pairs)
            
            logger.info(f"Loaded {len(self.qa_cache)} Q&A pairs into cache")
            
        except Exception as e:
            logger.error(f"Error loading Q&A documents: {e}")
    
    def _extract_qa_pairs(self, content: str) -> list:
        """Extract Q&A pairs from document content."""
        qa_pairs = []
        
        # Split content by numbered items (like "2.", "3.", etc.)
        # This handles the format where each Q&A block is numbered
        sections = re.split(r'\n\d+\.\s+', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Look for "Answer:" to split question and answer
            if 'Answer:' in section or 'answer:' in section:
                parts = re.split(r'(?i)\nanswer:\s*\n?', section, maxsplit=1)
                
                if len(parts) == 2:
                    questions_text = parts[0]
                    answer_text = parts[1]
                    
                    # Extract all questions from the questions section
                    questions = re.findall(r'"([^"]+\?)"', questions_text)
                    if not questions:
                        # Try without quotes
                        questions = [q.strip() for q in questions_text.split('\n') if '?' in q]
                    
                    # Clean answer text - stop at next numbered item or next question
                    answer_text = re.split(r'\n\d+\.\s+|^\d+\.\s+', answer_text)[0]
                    answer_text = answer_text.strip().strip('"')
                    
                    # Add Q&A pair for each question
                    for question in questions:
                        question = question.strip().strip('"\'')
                        if question and answer_text:
                            qa_pairs.append({
                                'question': question,
                                'answer': answer_text,
                                'normalized_q': normalize_text(question)
                            })
        
        return qa_pairs
    
    def get_exact_answer(self, question: str, similarity_threshold: float = 0.6) -> Optional[str]:
        """
        Get exact answer if question matches a Q&A pair.
        
        Args:
            question: User's question
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Exact answer string if match found, None otherwise
        """
        if not self.qa_cache:
            return None
        
        normalized_question = normalize_text(question)
        
        best_match = None
        best_score = 0
        
        for qa in self.qa_cache:
            normalized_qa = qa['normalized_q']
            
            # Direct substring match
            if normalized_qa in normalized_question or normalized_question in normalized_qa:
                score = len(normalized_qa) if len(normalized_qa) > 0 else 0
                if score > best_score:
                    best_score = score
                    best_match = qa
            
            # Token overlap match
            if not best_match:
                qa_tokens = set(normalized_qa.split())
                question_tokens = set(normalized_question.split())
                
                # Filter out very short tokens
                qa_tokens_filtered = {t for t in qa_tokens if len(t) > 3}
                question_tokens_filtered = {t for t in question_tokens if len(t) > 3}
                
                if qa_tokens_filtered and question_tokens_filtered:
                    overlap = len(qa_tokens_filtered & question_tokens_filtered)
                    token_score = overlap / len(qa_tokens_filtered)
                    
                    if token_score >= similarity_threshold and token_score > best_score:
                        best_score = token_score
                        best_match = qa
        
        if best_match and best_score > 0:
            logger.info(f"Q&A match found for '{question[:50]}...' with score {best_score:.2f}")
            return best_match['answer']
        
        return None

