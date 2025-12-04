#!/usr/bin/env python3
"""Verify all Q&A answers are exact"""

from qa_matcher import QAMatcher
import json

matcher = QAMatcher()

# Test questions from the user's list
test_cases = [
    ("What is Franquicia Boost?", "Franquicia Boost is a platform that connects franchisors, franchisees, consultants across Canada. It is the online franchise ecosystem where you can apply online, track your application and get the support you need at every step."),
    ("How do I find franchise opportunities here?", '"You can browse the franchise listings on our website in Franchise Opportunities under Explore Franchise Tab where you can search by industry, investment or location. Once something interests you, you can open the listing, see the details and apply directly from the same page in the Start/Continue Your Application."'),
    ("Are all franchises verified?", "Yes. Every franchise listed on Franquicia Boost goes through a verification check before being published. The goal is to help you explore real, trustworthy opportunities."),
    ("Does Franquicia Boost charge any fees?", None),  # Will check what we get
    ("What industries do you cover?", None),
    ("Is Franquicia Boost a marketplace or a consultancy?", None),
    ("Can I withdraw my application?", None),
    ("Do you offer free franchise consultation?", None),
]

print(f"Total Q&A pairs loaded: {len(matcher.qa_cache)}\n")
print("=" * 80)
print("VERIFICATION RESULTS")
print("=" * 80)

for question, expected_answer in test_cases:
    actual_answer = matcher.get_exact_answer(question)
    
    if actual_answer:
        # Check if it matches expected (if we have expected)
        if expected_answer:
            matches = actual_answer.strip() == expected_answer.strip()
            status = "✅ EXACT MATCH" if matches else "⚠️  MISMATCH"
        else:
            status = "✅ FOUND"
        
        print(f"\n{status}")
        print(f"Q: {question}")
        print(f"A: {actual_answer[:150]}..." if len(actual_answer) > 150 else f"A: {actual_answer}")
    else:
        print(f"\n❌ NO MATCH")
        print(f"Q: {question}")
        print(f"A: NOT FOUND IN Q&A CACHE")
    
    print("-" * 80)

# Show all questions in cache
print("\n" + "=" * 80)
print(f"ALL {len(matcher.qa_cache)} QUESTIONS IN CACHE:")
print("=" * 80)
for i, qa in enumerate(matcher.qa_cache, 1):
    print(f"{i:2d}. {qa['question']}")

