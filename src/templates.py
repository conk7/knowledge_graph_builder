PROMPT_TEMPLATE_FOR_LINKING = """You are a highly specialized API endpoint that only returns JSON.
Your task is to analyze the semantic link from Document A to Document B.
Your entire response MUST be a single, valid JSON object and nothing else.

Follow these formatting instructions precisely:
{format_instructions}

The possible link types are: {relation_types}

---
Document A: {text_a}
---
Document B: {text_b}
---

Now, perform the analysis and return the JSON object."""

PROMPT_TEMPLATE_FOR_RELEVANCE_CHECK = """You are a relevance analysis expert. Your task is to determine if a meaningful, 
non-trivial semantic link exists between Document A and Document B.
A trivial link is one of just "similarity". A meaningful link could be {link_types}, etc.
Think step-by-step and then conclude your answer with a single word: "Yes" or "No".

{format_instructions}

Document A:
---
{text_a}
---

Document B:
---
{text_b}
---
Step-by-step thought process:
1. What is the main topic of Document A?
2. What is the main topic of Document B?
3. Do they discuss the same concepts from different perspectives, or are they just vaguely related?
4. Is there a clear, definable link (like example, explanation, contradiction)?

Conclusion (Yes or No):"""
