#!/bin/bash
cd /home/ubuntu/FB-Bot_Python
python3 << 'PYTHON_EOF'
from qa_matcher import QAMatcher

matcher = QAMatcher()
print(f"Loaded {len(matcher.qa_cache)} Q&A pairs")

test_questions = [
    "What is Franquicia Boost?",
    "How do I find franchise opportunities here?",
    "Are all franchises verified?",
    "How do I apply for a franchise?",
    "Can I track my franchise application?"
]

for q in test_questions:
    answer = matcher.get_exact_answer(q)
    status = "✅" if answer else "❌"
    print(f"{status} {q}")
    if answer:
        print(f"   {answer[:80]}...")
PYTHON_EOF

