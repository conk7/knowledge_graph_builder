import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy

from .config import ExtractConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Prepositions and particles that significantly affect meaning.
MEANINGFUL_FUNCTION_WORDS = frozenset(
    {
        "без",
        "не",
        "ни",
        "с",
        "по",
        "для",
        "из",
        "от",
        "до",
        "под",
        "над",
        "через",
        "между",
        "против",
        "вне",
        "при",
        "without",
        "not",
        "with",
        "by",
        "according to",
        "for",
        "from",
        "of",
        "before",
        "until",
        "under",
        "over",
        "through",
        "between",
        "against",
        "outside",
        "at",
        "during",
    }
)


def load_nlp(lang: str = "ru"):
    model_map = {
        "ru": "ru_core_news_sm",
        "en": "en_core_web_sm",
    }
    model_name = model_map.get(lang, "ru_core_news_sm")
    try:
        return spacy.load(model_name)
    except OSError:
        logging.error(
            f"Model '{model_name}' not found. Please install it with 'python -m spacy download {model_name}'"
        )
        raise


def load_cross_encoder(model_name: str):
    """Lazily load CrossEncoder only when needed."""
    from sentence_transformers import CrossEncoder

    logging.info(f"Loading cross-encoder model: {model_name}...")
    model = CrossEncoder(model_name)
    logging.info("Cross-encoder model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Lemma extraction
# ---------------------------------------------------------------------------


def _extract_lemmas_and_head_from_doc(
    doc, keep_prepositions: bool = False
) -> Tuple[List[str], Optional[str]]:
    """
    Extracts significant lemmas and head word from a pre-processed spaCy Doc.
    If keep_prepositions is True, tokens whose lemma is in MEANINGFUL_FUNCTION_WORDS
    are kept alongside the standard POS tags (NOUN, ADJ, VERB, PROPN).
    """
    significant_pos = {"NOUN", "ADJ", "VERB", "PROPN"}
    meaningful_tokens = [w for w in doc if w.pos_ in significant_pos]

    if keep_prepositions:
        preposition_tokens = [
            w
            for w in doc
            if w.lemma_.lower() in MEANINGFUL_FUNCTION_WORDS
            and w not in meaningful_tokens
        ]
        meaningful_tokens.extend(preposition_tokens)
        # Restore original token order
        meaningful_tokens.sort(key=lambda w: w.i)

    # Fallback to all alpha tokens if no significant ones found
    if not meaningful_tokens:
        meaningful_tokens = [w for w in doc if w.is_alpha]

    lemmas = [w.lemma_.lower() for w in meaningful_tokens]

    head_lemma = None
    roots = [w for w in doc if w.dep_ == "ROOT" or w.head == w]
    if roots:
        head_lemma = roots[0].lemma_.lower()
        # If head lemma is a stop word, fallback to first noun or first meaningful word
        if head_lemma not in lemmas and meaningful_tokens:
            nouns = [w for w in meaningful_tokens if w.pos_ in {"NOUN", "PROPN"}]
            if nouns:
                head_lemma = nouns[0].lemma_.lower()
            else:
                head_lemma = meaningful_tokens[0].lemma_.lower()

    return lemmas, head_lemma


def extract_lemmas_and_head(
    phrase: str, nlp_model, keep_prepositions: bool = False
) -> Tuple[List[str], Optional[str]]:
    """
    Extracts significant lemmas (NOUN, ADJ, VERB, PROPN) from a phrase
    and identifies the head word.
    """
    phrase = phrase.replace("_", " ")
    doc = nlp_model(phrase)
    return _extract_lemmas_and_head_from_doc(doc, keep_prepositions=keep_prepositions)


def batch_extract_lemmas_and_heads(
    phrases: List[str], nlp_model, keep_prepositions: bool = False
) -> List[Tuple[List[str], Optional[str]]]:
    """
    Batch version of extract_lemmas_and_head using nlp.pipe() for better performance.
    """
    cleaned = [p.replace("_", " ") for p in phrases]
    return [
        _extract_lemmas_and_head_from_doc(doc, keep_prepositions=keep_prepositions)
        for doc in nlp_model.pipe(cleaned)
    ]


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------


def check_match_with_proximity(
    target_lemmas: List[str],
    target_head_lemma: Optional[str],
    sent_lemma_seq: List[str],
) -> bool:
    """
    Checks if target_lemmas closely match inside sent_lemma_seq within a specific window limit.
    Requires 100% of lemmas if target has <=2 words, and at least 75% for 3+ words.
    Also requires the head word to be present.
    """
    if not target_lemmas:
        return False

    if target_head_lemma and target_head_lemma not in sent_lemma_seq:
        return False

    total = len(target_lemmas)
    required = total if total <= 2 else math.ceil(total * 0.75)

    target_set = set(target_lemmas)

    # Indices in the sentence where any target lemma appears
    match_indices = [i for i, lemma in enumerate(sent_lemma_seq) if lemma in target_set]

    # Fast exit
    if len(set(sent_lemma_seq[i] for i in match_indices)) < required:
        return False

    # Linear sliding window: maximum span of 15 tokens between first and last match
    end = 0
    for start in range(len(match_indices)):
        # Advance end pointer to include all indices within the 15-token window
        while (
            end < len(match_indices) and match_indices[end] - match_indices[start] <= 15
        ):
            end += 1
        # Check uniqueness in window [start, end)
        unique_lemmas = set(sent_lemma_seq[k] for k in match_indices[start:end])
        if len(unique_lemmas) >= required and (
            not target_head_lemma or target_head_lemma in unique_lemmas
        ):
            return True

    return False


def check_match_with_cross_encoder(
    cross_encoder,
    link_phrase: str,
    sentences_info: List[Dict[str, Any]],
    threshold: float,
) -> Optional[Tuple[str, float]]:
    """
    Uses a CrossEncoder reranker to find the best matching sentence for a link phrase.
    Returns (sentence_text, score) if best score >= threshold, otherwise None.
    """
    if not sentences_info:
        return None

    sentence_texts = [s["text"] for s in sentences_info]
    pairs = [[link_phrase, sent] for sent in sentence_texts]

    scores = cross_encoder.predict(pairs, show_progress_bar=False)

    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        return sentence_texts[best_idx], best_score

    return None


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------


def process_text_sentences(text: str, nlp_model) -> List[Dict[str, Any]]:
    """
    Parses the text, returns a list of dictionaries with sentence text and sequence of lemmas.
    """
    doc = nlp_model(text)
    sentences_info = []
    for sent in doc.sents:
        sentences_info.append(
            {
                "text": sent.text.strip(),
                "lemmas": [w.lemma_.lower() for w in sent if w.is_alpha],
            }
        )
    return sentences_info


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_contexts(
    vault_path: Path,
    output_json: Path,
    config: ExtractConfig,
) -> None:
    """
    Extract contexts for all valid links in the markdown vault.
    Links not found in the main text body are ignored.
    """
    nlp = load_nlp(config.language)

    # Lazy-load cross-encoder only when needed
    cross_encoder = None
    if config.use_cross_encoder:
        cross_encoder = load_cross_encoder(config.cross_encoder_model)

    link_contexts: Dict[str, List[Dict[str, Any]]] = {}

    for file_path in vault_path.glob("*.md"):
        title = file_path.stem

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        parts = content.split(config.links_header)
        if len(parts) < 2:
            continue

        main_text = parts[0].strip()
        links_block = parts[1].strip()

        # Strip extraneous sections (См. также, Литература, etc.)
        stop_sections = [
            "См. также",
            "Литература",
            "Примечания",
            "Ссылки",
            "See also",
            "References",
            "Notes",
            "External links",
            "Bibliography",
            "Further reading",
        ]
        min_idx = len(main_text)

        stop_pattern_raw = (
            r"(?mi)^(?:#{1,6}\s+|={1,6}\s*)?("
            + "|".join(map(re.escape, stop_sections))
            + r")(?:\s*[:\-])?\s*$"
        )
        stop_regex = re.compile(stop_pattern_raw)

        for match in stop_regex.finditer(main_text):
            idx = match.start()
            if idx < min_idx:
                min_idx = idx

        if min_idx < len(main_text):
            logging.debug(
                f"Текст обрезан на секции: {main_text[min_idx : min_idx + 20]}..."
            )
            main_text = main_text[:min_idx].strip()

        # Pre-evaluate sentences from main text
        sentences_info = process_text_sentences(main_text, nlp)

        # Extract links (Obsidian format: [[Target]] or [[Target|Alias]])
        links = re.findall(r"\[\[(.*?)\]\]", links_block)

        # Parse link targets and aliases
        parsed_links = []
        for link_str in links:
            if "|" in link_str:
                target, alias = link_str.split("|", 1)
            else:
                target, alias = link_str, None
            parsed_links.append((target, alias))

        # Pre-compute lemmas for proximity matching
        phrases_to_process: List[str] = []
        phrase_to_idx: Dict[str, int] = {}
        phrase_lemmas: Dict[str, Tuple[List[str], Optional[str]]] = {}

        for target, alias in parsed_links:
            for phrase in [alias, target] if alias and alias != target else [target]:
                if phrase not in phrase_to_idx:
                    phrase_to_idx[phrase] = len(phrases_to_process)
                    phrases_to_process.append(phrase)

        if not config.use_cross_encoder:
            batch_results = batch_extract_lemmas_and_heads(
                phrases_to_process, nlp, keep_prepositions=config.keep_prepositions
            )
            phrase_lemmas = {
                phrase: batch_results[idx] for phrase, idx in phrase_to_idx.items()
            }

        # Match links against sentences
        valid_links = []
        for target, alias in parsed_links:
            context = None
            score = None
            candidates = [alias, target] if alias and alias != target else [target]

            if config.use_cross_encoder:
                # Run all candidates through cross-encoder, pick best score
                best_score = -1.0
                for phrase in candidates:
                    result = check_match_with_cross_encoder(
                        cross_encoder,
                        phrase,
                        sentences_info,
                        config.cross_encoder_threshold,
                    )
                    if result and result[1] > best_score:
                        context, score = result
                        best_score = result[1]
            else:
                # Proximity matching: try alias first, then target
                for phrase in candidates:
                    lemmas, head = phrase_lemmas[phrase]
                    for sent_info in sentences_info:
                        if check_match_with_proximity(
                            lemmas, head, sent_info["lemmas"]
                        ):
                            context = sent_info["text"]
                            break
                    if context:
                        break

            if context:
                valid_links.append(
                    {
                        "target": target,
                        "context": context,
                        "score": score,
                        "relation_type": None,  # To be filled by classifier
                    }
                )

        if valid_links:
            link_contexts[title] = valid_links

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(link_contexts, f, ensure_ascii=False, indent=4)

    total_links = sum(len(v) for v in link_contexts.values())
    logging.info(
        f"Extraction complete. Found contexts for {total_links} links across {len(link_contexts)} files."
    )
    logging.info(f"Results saved to {output_json}")
