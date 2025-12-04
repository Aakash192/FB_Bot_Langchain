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
            
            # Filter for Q&A documents and process ALL chunks
            qa_docs_found = 0
            for i, (content, metadata) in enumerate(zip(all_items['documents'], all_items['metadatas'])):
                doc_name = metadata.get('document_name', '')
                if any(keyword in doc_name for keyword in ['Knowledge', 'AI Bot', 'training', 'Question']):
                    # Parse Q&A pairs from content (may have multiple questions per chunk)
                    qa_pairs = self._extract_qa_pairs(content, metadata)
                    self.qa_cache.extend(qa_pairs)
                    qa_docs_found += 1
            
            logger.info(f"Processed {qa_docs_found} Q&A chunks, loaded {len(self.qa_cache)} Q&A pairs into cache")
            
        except Exception as e:
            logger.error(f"Error loading Q&A documents: {e}")
    
    def _extract_qa_pairs(self, content: str, metadata: Dict[str, Any]) -> list:
        """Extract Q&A pairs from document content."""
        qa_pairs = []
        
        # Format 1: Q1: Question / A: Answer format (Knowledge Question shared by Sana)
        qa_pattern1 = re.findall(r'Q\d+:\s*([^\n]+)\s*\n\s*A:\s*([^\n]+(?:\n(?!Q\d+:)[^\n]+)*)', content, re.MULTILINE)
        if qa_pattern1:
            for question, answer in qa_pattern1:
                question = question.strip()
                answer = answer.strip()
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'normalized_q': normalize_text(question)
                    })
        
        # Format 2: "Question?" / Answer: format (AI Bot Knowledge training)
        # Extract questions from content (could be quoted)
        questions_in_content = re.findall(r'"([^"]+\?)"', content)
        
        # Also check if section contains the question
        section = metadata.get('section', '')
        if section and section != 'N/A':
            # Remove numeric prefix like "16. " or "17: "
            section_clean = re.sub(r'^\d+\.?\s*:?\s*', '', section).strip().strip('"\'')
            if section_clean and '?' in section_clean:
                questions_in_content.append(section_clean)
        
        # Extract answer from content
        answer_text = None
        
        if 'Answer:' in content or 'answer:' in content:
            # Split by "Answer:" marker
            parts = re.split(r'(?i)\nanswer:\s*\n?', content, maxsplit=1)
            if len(parts) == 2:
                answer_text = parts[1]
                # Clean answer - stop at next numbered item or next question
                answer_text = re.split(r'\n\d+\.\s+|\nQ\d+:', answer_text)[0]
                answer_text = answer_text.strip().strip('"')
        
        # If no explicit "Answer:" found, use the content after the question
        if not answer_text and questions_in_content:
            # Remove the question part and take what's left
            for q in questions_in_content:
                if q in content:
                    parts = content.split(q, 1)
                    if len(parts) == 2:
                        answer_text = parts[1].strip()
                        # Clean answer
                        answer_text = re.split(r'\n\d+\.\s+|\nQ\d+:', answer_text)[0]
                        answer_text = answer_text.strip().strip('"')
                        break
        
        # Create Q&A pairs for format 2
        if questions_in_content and answer_text:
            for question in questions_in_content:
                question = question.strip().strip('"\'')
                if question:
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

