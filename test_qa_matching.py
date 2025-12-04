#!/usr/bin/env python3
"""Test Q&A matching"""

from qa_matcher import QAMatcher

matcher = QAMatcher()

print(f"Loaded {len(matcher.qa_cache)} Q&A pairs\n")

# Show all questions in cache
print("Questions in cache:")
for i, qa in enumerate(matcher.qa_cache[:20], 1):
    print(f"{i}. {qa['question'][:80]}...")

# Test queries
test_queries = [
    "What is Franquicia Boost?",
    "What services does Franquicia Boost offer?",
    "Who can use Franquicia Boost",
    "How do I find franchise opportunities?"
]

print("\n" + "=" * 80)
print("TESTING QUERIES")
print("=" * 80)

for query in test_queries:
    answer = matcher.get_exact_answer(query, similarity_threshold=0.5)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer if answer else 'NO MATCH'}")
    print("-" * 80)

