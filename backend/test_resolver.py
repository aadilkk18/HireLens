from role_resolver import resolve_role

test_inputs = [
    "I want to build APIs and work with databases",
    "something with AI and machine learning",
    "I want to design app interfaces and user flows",
    "network stuff and fixing computers",
    "Junior Software Engineer",
]

for text in test_inputs:
    result = resolve_role(text)
    print(f"Input: '{text}' -> Matched: {result['matched_role']} (similarity: {result['similarity']}, {result['source']})")