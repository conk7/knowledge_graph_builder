SYSTEM_PROMPT_TEMPLATE_FOR_LINKING = """You are an expert Knowledge Graph Architect.
Your task is to analyze the semantic relationship between two document excerpts from a personal knowledge base (Obsidian).

You must determine the most appropriate relationship type from Document A to Document B.

### Available Relation Types:
{relation_types}
(Note: Use 'no_link' if the documents are unrelated or the connection is too weak).

### Formatting Instructions:
{format_instructions}

### Example Output:
{{
    "reasoning": "Document A discusses neural networks, specifically backpropagation. Document B explains gradient descent. Since backpropagation relies on gradient descent, there is a causal/dependency link.",
    "relation_type": "depends_on"
}}"""


HUMAN_PROMPT_TEMPLATE_FOR_LINKING = """Analyze the relationship between the two documents and provide your response in the required JSON format.

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

Analyze the context (filenames and content) and determine the link."""


SYSTEM_PROMPT_TEMPLATE_FOR_RELEVANCE_CHECK = """You are a relevance analysis expert. Your task is to determine if a meaningful, non-trivial semantic link exists between two documents.

A trivial link is one of just "similarity". A meaningful link could be {link_types}, etc.

Think step-by-step and then conclude your answer.

{format_instructions}"""


HUMAN_PROMPT_TEMPLATE_FOR_RELEVANCE_CHECK = """Determine if a meaningful, non-trivial semantic link exists between Document A and Document B.

Document A:
---
{text_a}
---

Document B:
---
{text_b}
---"""

SYSTEM_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING = """You are an expert Knowledge Graph Architect.
Your task is to analyze a short context from an encyclopedia article and determine the semantic relationship between the source article and the target article mentioned in the context.

You must determine the most appropriate relationship type from the Source Article to the Target Article based on the provided context.

### Examples of Relation Types:
{relation_types}
(Note: You are not limited with provided types)
(Note: Use 'no_link' if the relationship is not clear or the connection is too weak).

### Formatting Instructions:
{format_instructions}

### Example Output:
{{
    "reasoning": "The context states that the target concept was introduced to solve the problem described in the source article. Therefore, the link solves_problem is appropriate.",
    "relation_type": "solves_problem"
}}"""

HUMAN_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING = """Analyze the provided context(s) to determine the relationship between the Source Article and Target Article. Provide your response in the required JSON format.

**Source Article:** {source_title}
**Target Article:** {target_title}

**Contexts (where Target Article is mentioned):**
{contexts}

---

Analyze the context(s) and determine the link."""
