#!/usr/bin/env python3
from qa_matcher import QAMatcher

matcher = QAMatcher()
print(f'Total Q&A pairs: {len(matcher.qa_cache)}\n')

# Find service-related questions
services_q = [q for q in matcher.qa_cache if 'service' in q['normalized_q'].lower()]
print(f'Service-related Q&As: {len(services_q)}')
for q in services_q:
    print(f"  Q: {q['question']}")
    print(f"  A: {q['answer'][:100]}...")
    print()

# Find "who can use" questions
who_q = [q for q in matcher.qa_cache if 'who' in q['normalized_q'].lower()]
print(f'\n"Who" questions: {len(who_q)}')
for q in who_q:
    print(f"  Q: {q['question']}")
    print(f"  A: {q['answer'][:100]}...")
    print()

