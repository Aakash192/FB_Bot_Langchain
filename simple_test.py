from qa_matcher import QAMatcher

matcher = QAMatcher()
answer = matcher.get_exact_answer("What is Franquicia Boost?")
print("Answer:", answer)
print("\nExpected: Franquicia Boost is a platform that connects franchisors, franchisees, consultants across Canada.")
print("Match:", "platform that connects franchisors" in answer)

