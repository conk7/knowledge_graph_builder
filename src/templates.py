PROMPT_TEMPLATE_FOR_LINKING = """You are an expert Knowledge Graph Architect. 
Your task is to analyze the semantic relationship between two document excerpts from a personal knowledge base (Obsidian).

You must determine the most appropriate relationship type from Document A to Document B.

### Available Relation Types:
{relation_types}
(Note: Use 'no_link' if the documents are unrelated or the connection is too weak).

### Formatting Instructions:
Return ONLY a valid JSON object. Do not add any markdown formatting (like ```json).
{format_instructions}

### Example Output:
{{
    "reasoning": "Document A discusses neural networks, specifically backpropagation. Document B explains gradient descent. Since backpropagation relies on gradient descent, there is a causal/dependency link.",
    "relation_type": "depends_on"
}}

---

### Document A
**Filename:** {filename_a}
**Content:**
"{text_a}"

---

### Document B
**Filename:** {filename_b}
**Content:**
"{text_b}"

---

Analyze the context (filenames and content) and determine the link.
"""

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
