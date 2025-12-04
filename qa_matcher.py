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
            
            # Separate Q&A documents by priority
            # Priority 1: "AI Bot Knowledge training" (user's primary Q&A source)
            # Priority 2: "Knowledge Question shared by Sana" (secondary/fallback)
            priority_qa = []
            secondary_qa = []
            
            for i, (content, metadata) in enumerate(zip(all_items['documents'], all_items['metadatas'])):
                doc_name = metadata.get('document_name', '')
                
                # Parse Q&A pairs
                qa_pairs = None
                if 'AI Bot' in doc_name or 'training' in doc_name:
                    qa_pairs = self._extract_qa_pairs(content, metadata)
                    if qa_pairs:
                        priority_qa.extend(qa_pairs)
                elif 'Knowledge' in doc_name or 'Question' in doc_name:
                    qa_pairs = self._extract_qa_pairs(content, metadata)
                    if qa_pairs:
                        secondary_qa.extend(qa_pairs)
            
            # Add priority Q&A first, then secondary (avoids duplicates being overridden)
            # Use dict to deduplicate by normalized question, keeping first occurrence (priority)
            qa_dict = {}
            for qa in priority_qa:
                qa_dict[qa['normalized_q']] = qa
            
            # Debug: Check for key questions
            test_q = normalize_text("What is Franquicia Boost?")
            if test_q in qa_dict:
                logger.info(f"Priority Q&A: 'What is Franquicia Boost?' = {qa_dict[test_q]['answer'][:50]}...")
            
            for qa in secondary_qa:
                # Only add if not already in dict (priority takes precedence)
                if qa['normalized_q'] not in qa_dict:
                    qa_dict[qa['normalized_q']] = qa
                else:
                    logger.debug(f"Skipping duplicate from secondary: {qa['question'][:50]}...")
            
            self.qa_cache = list(qa_dict.values())
            
            logger.info(f"Loaded {len(self.qa_cache)} Q&A pairs (Priority: {len(priority_qa)}, Secondary: {len(secondary_qa)})")
            
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
            # If we found Q1:/A: format, return now (don't try other formats)
            if qa_pairs:
                return qa_pairs
        
        # Format 2: "Question?" / Answer: format (AI Bot Knowledge training)
        # This format has numbered sections (2., 3., 4., etc.)
        # The first section (before "2.") may have multiple questions sharing one answer
        
        # Split by numbered sections starting from 2 onwards
        parts = re.split(r'\n(\d+)\.\s+', content)
        
        # Process the first part (before any numbering - contains questions 1-2)
        if len(parts) > 0:
            first_section = parts[0]
            questions = re.findall(r'"([^"]+\?)"', first_section)
            
            if 'Answer:' in first_section:
                answer_parts = re.split(r'Answer:\s*\n', first_section, maxsplit=1)
                if len(answer_parts) == 2:
                    answer_text = answer_parts[1].strip().strip('"')
                    
                    # All questions in first section share this answer
                    for question in questions:
                        question = question.strip()
                        if question and answer_text:
                            qa_pairs.append({
                                'question': question,
                                'answer': answer_text,
                                'normalized_q': normalize_text(question)
                            })
        
        # Process numbered sections (2 onwards)
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
            
            number = parts[i]  # The captured number
            section_content = parts[i + 1]  # The content after the number
            
            # Extract quoted questions
            questions = re.findall(r'"([^"]+\?)"', section_content)
            
            # Extract answer (text after "Answer:")
            if 'Answer:' in section_content:
                answer_parts = re.split(r'Answer:\s*\n', section_content, maxsplit=1)
                if len(answer_parts) == 2:
                    answer_text = answer_parts[1]
                    # Stop at next section (don't include it)
                    answer_text = re.split(r'\n\d+\.\s+', answer_text)[0]
                    answer_text = answer_text.strip().strip('"')
                    
                    # Add all questions with this answer
                    for question in questions:
                        question = question.strip()
                        if question and answer_text:
                            qa_pairs.append({
                                'question': question,
                                'answer': answer_text,
                                'normalized_q': normalize_text(question)
                            })
        
        # Also check section field for single Q&A chunks
        section = metadata.get('section', '')
        if section and section != 'N/A' and not qa_pairs:
            # Remove numeric prefix
            section_clean = re.sub(r'^\d+\.?\s*:?\s*', '', section).strip().strip('"\'')
            if section_clean and '?' in section_clean:
                # Extract answer from content
                if 'Answer:' in content or 'answer:' in content:
                    parts = re.split(r'(?i)\nanswer:\s*\n?', content, maxsplit=1)
                    if len(parts) == 2:
                        answer_text = parts[1].strip().strip('"')
                        qa_pairs.append({
                            'question': section_clean,
                            'answer': answer_text,
                            'normalized_q': normalize_text(section_clean)
                        })
                else:
                    # Answer might be the content itself
                    answer_text = content.replace(section_clean, '').strip()
                    if answer_text:
                        qa_pairs.append({
                            'question': section_clean,
                            'answer': answer_text,
                            'normalized_q': normalize_text(section_clean)
                        })
        
        return qa_pairs
    
    def get_exact_answer(self, question: str, similarity_threshold: float = 0.7) -> Optional[str]:
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
        question_tokens = set(normalized_question.split())
        
        best_match = None
        best_score = 0
        best_method = None
        
        for qa in self.qa_cache:
            normalized_qa = qa['normalized_q']
            qa_tokens = set(normalized_qa.split())
            
            # Method 1: Exact match (highest priority)
            if normalized_qa == normalized_question:
                return qa['answer']
            
            # Method 2: One is complete substring of the other (high priority)
            # But require significant overlap to avoid false matches
            if normalized_qa in normalized_question:
                # Check if the question is mostly about the Q&A topic
                # e.g., "what is franquicia boost" matches "what is franquicia boost"
                # but "what is venture x" should NOT match "what is an fdd"
                score = len(normalized_qa) / len(normalized_question)
                if score >= 0.7 and len(normalized_qa) > best_score:  # At least 70% overlap
                    best_score = len(normalized_qa)
                    best_match = qa
                    best_method = "substring"
            elif normalized_question in normalized_qa:
                score = len(normalized_question) / len(normalized_qa)
                if score >= 0.7 and len(normalized_question) > best_score:
                    best_score = len(normalized_question)
                    best_match = qa
                    best_method = "substring"
            
            # Method 3: Token overlap (STRICT - only for very similar questions)
            # Filter out very short and generic tokens
            stopwords = {'what', 'does', 'this', 'that', 'with', 'from', 'have', 'here', 'about', 'your', 'the'}
            qa_tokens_filtered = {t for t in qa_tokens if len(t) > 3 and t not in stopwords}
            question_tokens_filtered = {t for t in question_tokens if len(t) > 3 and t not in stopwords}
            
            if qa_tokens_filtered and question_tokens_filtered:
                overlap = len(qa_tokens_filtered & question_tokens_filtered)
                overlap_tokens = qa_tokens_filtered & question_tokens_filtered
                
                # STRICT REQUIREMENT: Questions must share specific subject tokens
                # e.g., "franquicia boost", not just generic "franchise" or "fdd"
                # This prevents "what is venture x" from matching "what is an fdd"
                
                # Check if they share the SAME key noun phrases
                # For franchise questions, check if franchise name matches
                has_franchise_name_qa = any(t in qa_tokens for t in ['venture', 'anago', 'snapon', 'tint', 'world', 'rnr'])
                has_franchise_name_q = any(t in question_tokens for t in ['venture', 'anago', 'snapon', 'tint', 'world', 'rnr'])
                
                # If Q&A is about a specific franchise, question must also be about that franchise
                if has_franchise_name_qa and not has_franchise_name_q:
                    continue  # Don't match
                
                # If question is about a specific franchise, don't match generic Q&A
                if has_franchise_name_q and not has_franchise_name_qa:
                    continue  # Don't match
                
                # Require at least 80% of Q&A tokens to be in the question (very strict)
                if len(qa_tokens_filtered) > 0:
                    coverage = overlap / len(qa_tokens_filtered)
                    
                    # Only match if almost all Q&A tokens are covered
                    if coverage >= 0.8 and overlap >= 2:  # At least 80% coverage and 2+ matching tokens
                        if coverage > best_score:
                            best_score = coverage
                            best_match = qa
                            best_method = "token"
        
        if best_match and best_score > 0:
            logger.info(f"Q&A match found for '{question[:50]}...' using {best_method} method with score {best_score:.2f}")
            return best_match['answer']
        
        return None

