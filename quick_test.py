from qa_matcher import QAMatcher

m = QAMatcher()
tests = [
    ("What is Franquicia Boost?", True),
    ("what is venture x about?", False),
    ("what is anago about?", False),
    ("How do I find franchise opportunities?", True),
    ("Are all franchises verified?", True)
]

for q, should_match in tests:
    ans = m.get_exact_answer(q)
    matched = ans is not None
    status = "✅" if matched == should_match else "❌"
    print(f"{status} '{q}' - Match={matched}, Expected={should_match}")
    if ans:
        print(f"   Answer: {ans[:60]}...")

