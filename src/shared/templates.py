SYSTEM_PROMPT_TEMPLATE_FOR_LINKING = """You are an expert Knowledge Graph Architect.
Your task is to extract the DIRECTED semantic relationship between two entities from a personal knowledge base (Obsidian).

CRITICAL: The core entities are the FILENAMES. The provided text content is only the evidence. 
You must determine the relationship strictly in ONE DIRECTION: 
[Document A (Source)] ---> [Document B (Target)]

### Available Relation Types:
{relation_types}

### IMPORTANT RULES:
1. STRICT DIRECTION: Answer the question "How does [Document A] relate to [Document B]?". Do NOT extract the reverse relationship.
2. EXACT MATCH: You MUST choose EXACTLY ONE relation type from the list above. Do not invent new types.
3. HIGH THRESHOLD: If the texts simply mention the same broad topic without a clear, specific connection between the two exact entities, choose 'No link'.
4. GRAPH RAG CONTEXT: Write a concise 1-2 sentence 'semantic_detail' that explains the exact historical, functional, or logical connection based ONLY on the provided text.

### Formatting Instructions:
{format_instructions}

### Example 1 (Positive Link):
{{
    "reasoning": "Document A discusses neural networks, specifically backpropagation. Document B explains gradient descent. Since backpropagation relies on gradient descent, there is a causal/dependency link.",
    "relation_type": "Depends on"
}}

### Example 2 (Negative/Weak Link):
{{
    "reasoning": "Both documents mention the Space Race, but there is no direct functional or chronological relationship established between them in the text.",
    "relation_type": "No link",
}}"""


HUMAN_PROMPT_TEMPLATE_FOR_LINKING = """Analyze the directed relationship: [Document A] ---> [Document B].
Extract the link based ONLY on the provided evidence.

### [Source Entity]: Document A
**Filename:** {filename_a}
**Content Evidence:**
"{text_a}"

---

### [Target Entity]: Document B
**Filename:** {filename_b}
**Content Evidence:**
"{text_b}"

---
Determine how [Document A] relates to [Document B]. Provide your response in valid JSON."""


SYSTEM_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING = """You are an expert Knowledge Graph Architect.
Your task is to analyze a short context from an encyclopedia article and determine the semantic relationship between the source article and the target article mentioned in the context.

You must determine the most appropriate relationship type from the Source Article to the Target Article based on the provided context.

### Examples of Relation Types:
{relation_types}
(Note: You are not limited with provided types)
(Note: Use 'No link' if the relationship is not clear or the connection is too weak).

### Formatting Instructions:
{format_instructions}

### Example Output:
{{
    "reasoning": "The context states that the target concept was introduced to solve the problem described in the source article. Therefore, the link solves_problem is appropriate.",
    "relation_type": "Solves problem"
}}"""

HUMAN_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING = """Analyze the provided context(s) to determine the relationship between the Source Article and Target Article. Provide your response in the required JSON format.

**Source Article:** {source_title}
**Target Article:** {target_title}

**Contexts (where Target Article is mentioned):**
{contexts}

---

Analyze the context(s) and determine the link."""


SYSTEM_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION = """You are an expert Knowledge Graph Architect.
Your task is to resolve conflicting relation predictions for a SINGLE DIRECTED edge in a graph.

CRITICAL DIRECTION:
[Document A (Source)] ---> [Document B (Target)]
You must answer only: "How does A relate to B?".
Never infer or output the reverse direction.

### Available Relation Types:
{relation_types}

### Input You Will Receive:
1. Directed file pair (A filename, B filename).
2. Multiple chunk-level candidate predictions for this same direction.
3. Evidence snippets for each candidate.

### Decision Rules:
1. Choose EXACTLY ONE final relation type from the list above, or 'No link'.
2. Prefer relation types supported by the strongest, most specific evidence.
3. If evidence is mixed, resolve using direction and semantic specificity.
4. If evidence is too weak/ambiguous for this direction, choose 'No link'.
5. Do not invent relation types.

### Output Requirements:
- Provide concise reasoning focused on direction and evidence quality.
- Return strict JSON only.

### Formatting Instructions:
{format_instructions}
"""


HUMAN_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION = """Resolve the final directed relation for:
[Document A] ---> [Document B]

### Source (A)
- Filename: {filename_a}

### Target (B)
- Filename: {filename_b}

### Candidate Chunk-Level Predictions
{candidate_predictions}

### Evidence
{evidence}

Return one final relation type for A -> B in valid JSON."""
