# Franquicia Boost AI Chatbot - Setup Complete

## What Was Fixed

### 1. OpenAI API v1.0+ Compatibility ✅
- Replaced `openai.Embedding` with `CustomOpenAIEmbeddingFunction`
- System now works with OpenAI API v1.0+

### 2. Q&A Exact Matching System ✅  
- Created `qa_matcher.py` for keyword-based exact answer matching
- Loads Q&A pairs from knowledge documents
- Returns exact answers (bypasses LLM) when question matches
- Falls back to RAG for FDD questions

### 3. WordPress Widget ✅
- File: `wpcode-widget.html`
- Self-contained (HTML + CSS + JavaScript)
- Matches your reference design
- Ready to paste into WPCode

### 4. CORS Fixed ✅
- Configured for `staging2.franquiciaboost.com`
- WordPress can now access the API

### 5. Code on GitHub ✅
- Repository: https://github.com/Aakash192/FB_Bot_Langchain.git

## Current Issue & Fix

**Problem**: Q&A matcher is too aggressive and matches FDD questions to Q&A answers

**Latest Fix Applied** (in `qa_matcher.py`):
- Stricter matching algorithm
- Filters out franchise-specific questions ("venture x", "anago") from matching generic Q&A
- Increased similarity threshold to 80%
- Added stopword filtering

**To Apply the Fix:**

```bash
cd /home/ubuntu/FB-Bot_Python
bash restart_service.sh
```

Then test:

```bash
# Should match Q&A
curl -X POST https://fqbbot.com/api/chat -H "Content-Type: application/json" \
  -d '{"question":"What is Franquicia Boost?"}'

# Should use RAG (NOT match Q&A)
curl -X POST https://fqbbot.com/api/chat -H "Content-Type: application/json" \
  -d '{"question":"what is venture x about?"}'
```

## Files Structure

```
FB-Bot_Python/
├── app.py                      # Flask app with CORS fixes
├── docx_rag_pipeline.py        # RAG pipeline with OpenAI v1.0+ and Q&A integration
├── qa_matcher.py               # Q&A exact matching system (UPDATED)
├── wpcode-widget.html          # WordPress widget
├── rebuild_vector_store.py     # Vector store rebuild script
├── WPCODE_INSTRUCTIONS.md      # WordPress setup guide
├── restart_service.sh          # Restart Gunicorn script (NEW)
├── requirements.txt            # Python dependencies
└── data/                       # Your documents (26 total)
```

## Quick Commands

```bash
# Restart the service
bash /home/ubuntu/FB-Bot_Python/restart_service.sh

# Check health
curl https://fqbbot.com/api/health

# Test Q&A
curl -X POST https://fqbbot.com/api/chat -H "Content-Type: application/json" \
  -d '{"question":"What is Franquicia Boost?"}'

# Test FDD question
curl -X POST https://fqbbot.com/api/chat -H "Content-Type: application/json" \
  -d '{"question":"what does anago do?"}'
```

## WordPress Integration

1. Open `wpcode-widget.html`
2. Copy all contents
3. In WordPress: Code Snippets > Add Snippet
4. Select Type: HTML
5. Paste the code
6. Activate and save

Widget will appear in bottom-right corner of your WordPress site.

## Support

For questions or issues:
- Check `FINAL_STATUS_AND_FIX.md` for detailed troubleshooting
- API logs: Check Gunicorn workers
- Q&A testing: Run `python3 quick_test.py`

