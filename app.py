"""
Flask web application for Franquicia Boost Assistant
Provides a web UI for the DOCX RAG Pipeline
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from docx_rag_pipeline import DocxExcelRagPipeline
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the RAG pipeline
pipeline = None

def clean_answer(text):
    """
    Clean answer text by removing special characters and formulas.
    
    Args:
        text: Raw answer text from the pipeline
        
    Returns:
        Cleaned text without special characters and formulas
    """
    if not text:
        return text
    
    # Remove formulas (patterns that look like formulas)
    # Excel-style formulas starting with = (like =SUM(A1:A10), =A1+B1, etc.)
    text = re.sub(r'=\s*[A-Z0-9+\-*/()\[\]:,]+', '', text)
    
    # Mathematical formulas with operators (like 5+3=8, 10/2=5, etc.)
    text = re.sub(r'\b\d+\s*[+\-*/=]\s*\d+\s*[+\-*/=]?\s*\d*\b', '', text)
    
    # Formulas in parentheses with operators (like (A+B), (x*y), etc.)
    text = re.sub(r'\([^)]*[+\-*/=][^)]*\)', '', text)
    
    # LaTeX-style formulas (between $ or $$)
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\$\$[^$]+\$\$', '', text)
    
    # Code-like expressions with brackets and operators
    text = re.sub(r'\[[^\]]*[+\-*/=][^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*[+\-*/=][^}]*\}', '', text)
    
    # Remove standalone special characters: #, /, \, |, <, >, {, }, [, ], ^, ~, `, @
    # But preserve them if they're part of words or common patterns
    # Remove # when it's standalone or at start of line
    text = re.sub(r'#+', '', text)
    
    # Remove / but preserve common patterns like dates (but user wants / removed)
    text = text.replace('/', ' ')
    
    # Remove markdown bold markers (**)
    text = text.replace('**', '')
    
    # Remove other special characters
    special_chars_to_remove = ['\\', '|', '<', '>', '{', '}', '[', ']', '^', '~', '`', '@']
    for char in special_chars_to_remove:
        text = text.replace(char, '')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up multiple periods or commas
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r',{2,}', ',', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def initialize_pipeline():
    """Initialize the RAG pipeline and load vector store."""
    global pipeline
    try:
        pipeline = DocxExcelRagPipeline(
            chroma_db_path="chroma_db",
            chunk_size=1000,
            retrieval_count=6,  # Increased from 3 to 6 for better context
            temperature=0.2,  # Lower temperature for more accurate, consistent answers
            max_response_tokens=1200  # Increased for more detailed answers
        )
        
        # Try to load existing vector store
        if pipeline.load_vector_store():
            stats = pipeline.get_collection_stats()
            logger.info(f"Loaded vector store with {stats.get('total_chunks', 0)} chunks")
        else:
            logger.warning("No existing vector store found. Please build it first.")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        return False

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400
        
        if pipeline is None:
            return jsonify({
                'success': False,
                'error': 'Pipeline not initialized. Please ensure vector store is built.'
            }), 500
        
        logger.info(f"Processing query: {question}")
        
        # Query the pipeline (same as terminal version)
        result = pipeline.query(question)
        
        # Clean the answer to remove special characters and formulas
        raw_answer = result.get('answer', 'No answer generated')
        cleaned_answer = clean_answer(raw_answer)
        
        return jsonify({
            'success': True,
            'answer': cleaned_answer,
            'sources': result.get('sources', []),
            'metadata': result.get('metadata', {})
        })
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if pipeline is None:
        return jsonify({
            'status': 'error',
            'message': 'Pipeline not initialized'
        }), 500
    
    try:
        stats = pipeline.get_collection_stats()
        return jsonify({
            'status': 'ok',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Initializing Franquicia Boost Assistant...")
    if initialize_pipeline():
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize pipeline. Exiting.")

