#!/bin/bash
cd /home/ubuntu/FB-Bot_Python

echo "Testing Q&A matching with stricter algorithm..."
python3 -c '
from qa_matcher import QAMatcher

matcher = QAMatcher()

tests = [
    "What is Franquicia Boost?",
    "what is venture x about?",
    "what is anago about?",
    "How do I find franchise opportunities?",
    "Are all franchises verified?"
]

for q in tests:
    answer = matcher.get_exact_answer(q)
    if answer:
        print(f"✅ {q}")
        print(f"   {answer[:80]}...")
    else:
        print(f"❌ {q} - NO MATCH (will use RAG)")
    print()
'

