import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import json_repair
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.kg_builder.config import DEFAULT_LINK_TYPES, RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)


DEFAULT_TOP_N = 5
DEFAULT_BATCH_SIZE = 1
DEFAULT_CONCURRENT_REQUESTS = 5

_LLM_MAX_RETRIES: int = 10
_LLM_TIMEOUT_SEC: int = 240


class RelationPrediction(BaseModel):
    id: int = Field(description="0-based index of the item within the current batch")
    reasoning: str = Field(description="One-sentence explanation")
    predicted_type: str = Field(
        description="Exactly one allowed relation type, or 'no link'"
    )


class BatchResponse(BaseModel):
    results: list[RelationPrediction]


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Ontology Engineer tasked with classifying semantic relationships between a source document (Subject) and a target entity (Object).

For each item, you will receive:
  - Source: The name of the subject document.
  - Entity: The name of the target object.
  - Context sentences extracted from the Source that mention the Entity.

Your job: decide which single relationship best describes how the Source relates to the Entity.

### ALLOWED RELATION TYPES
You must use EXACTLY ONE of the following types (case-insensitive). If no clear relationship exists, output "no link".
{relation_types}

### RELATION DEFINITIONS & HEURISTICS
- scheduled on: The Source (a task, event, or note) is planned for the date/time of the Entity.
- manufactures / develops: The Source creates, produces, builds, or designs the Entity.
- uses / requires: The Source depends on, utilizes, or needs the Entity to function.
- belongs to / is a / belongs to genus: Taxonomic, hierarchical, or categorical relationships.
- supersedes: The Source replaces or renders the Entity obsolete.
- contradicts: The Source opposes or disproves the Entity.
- incorporates: The Source includes the Entity as a component, part, or underlying principle.
- mentions: The Source refers to the Entity in passing or as an anecdote, without a strong structural, causal, or functional link. (Use this sparingly, only when no specific relationship applies).

### EXAMPLES

[0] Source: "Haircut Appointment"  →  Entity: "21 марта"
  Context sentences:
    1. Записался к мастеру на 11:00 21 марта.
Output:
{{"results": [{{"id": 0, "reasoning": "The source is an appointment that is planned to happen on the specified date.", "predicted_type": "scheduled on"}}]}}

[1] Source: "Rutherford model"  →  Entity: "Plum pudding model"
  Context sentences:
    1. The concept arose after Ernest Rutherford directed the Geiger–Marsden experiment in 1909, which showed much more alpha particle recoil than J. J. Thomson's plum pudding model of the atom could explain.
Output:
{{"results": [{{"id": 1, "reasoning": "The Rutherford model was created to replace the older plum pudding model because it explained phenomena the old model couldn't.", "predicted_type": "supersedes"}}]}}

[2] Source: "Isaac Newton"  →  Entity: "Apple"
  Context sentences:
    1. Newton often told the story that he was inspired to formulate his theory of gravitation by watching the fall of an apple from a tree.
Output:
{{"results": [{{"id": 2, "reasoning": "The apple is featured in a historical anecdote about Newton, with no deep systemic or taxonomic relationship.", "predicted_type": "mentions"}}]}}

[3] Source: "Semiconductor device fabrication"  →  Entity: "Integrated circuit"
  Context sentences:
    1. Semiconductor device fabrication is the process used to manufacture semiconductor devices, typically integrated circuits (ICs) such as microprocessors...
Output:
{{"results": [{{"id": 3, "reasoning": "The source describes a fabrication process that directly produces integrated circuits.", "predicted_type": "manufactures"}}]}}

Respond with ONLY a valid JSON object — no markdown fences, no extra text — matching this exact structure:
{{"results": [{{"id": int, "reasoning": "str", "predicted_type": "str"}}]}}
"""

HUMAN_PROMPT_TEMPLATE = """\
Classify the following {count} pair(s). Read the context carefully, formulate your reasoning, and then select the best matching relation type.

{items}
"""


def _load_llm() -> Any:
    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    model = os.environ.get("MODEL", "")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    top_p = float(os.environ.get("LLM_TOP_P", "0.1"))

    logger.info(f"Initialising LLM provider={provider!r} model={model!r}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            model_kwargs={"top_p": top_p},
            max_retries=_LLM_MAX_RETRIES,
            timeout=_LLM_TIMEOUT_SEC,
        )
        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception:
            pass
        return llm

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_retries=_LLM_MAX_RETRIES,
        )

    elif provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        api_key = os.environ.get("CEREBRAS_API_KEY")
        llm = ChatCerebras(
            model=model or "llama3.1-8b",
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_retries=_LLM_MAX_RETRIES,
            timeout=_LLM_TIMEOUT_SEC,
        )
        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception:
            pass
        return llm

    elif provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=_LLM_MAX_RETRIES,
            request_timeout=_LLM_TIMEOUT_SEC,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. "
            "Set LLM_PROVIDER to one of: openai, google, cerebras, groq."
        )


def _rank_sentences(
    cross_encoder: CrossEncoder,
    query: str,
    sentences_meta: list[dict],
    top_n: int,
) -> list[str]:
    if not sentences_meta:
        return []

    texts = [item["sentence"] for item in sentences_meta]
    scores = cross_encoder.predict([(query, t) for t in texts])

    scored = [
        (item["sentence"], bool(item.get("is_direct_mention")), float(s))
        for item, s in zip(sentences_meta, scores)
    ]
    scored.sort(key=lambda x: (not x[1], -x[2]))
    return [s[0] for s in scored[:top_n]]


def _load_gold(gold_dir: Path, vault_dir: Path) -> list[dict]:
    items: list[dict] = []

    for gold_file in sorted(gold_dir.glob("*.json")):
        sample_name = gold_file.stem
        vault_sample_dir = vault_dir / sample_name

        config_path = vault_sample_dir / ".kg_builder" / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                allowed_types = cfg.get("llm_link_types", DEFAULT_LINK_TYPES)
            except Exception as e:
                logger.warning(f"Could not read config for {sample_name}: {e}")
                allowed_types = DEFAULT_LINK_TYPES
        else:
            logger.warning(
                f"No .kg_builder/config.json for {sample_name}, using defaults."
            )
            allowed_types = DEFAULT_LINK_TYPES

        data: list[dict] = json.loads(gold_file.read_text(encoding="utf-8"))

        grouped: dict[tuple, dict] = {}
        for row in data:
            key = (row["source_file"], row["entity"])
            if key not in grouped:
                grouped[key] = {
                    "link_type": None,
                    "sentences_meta": [],
                    "allowed_types": allowed_types,
                    "sample_name": sample_name,
                }
            if row.get("link_type") is not None:
                grouped[key]["link_type"] = row["link_type"]
            grouped[key]["sentences_meta"].append(
                {
                    "sentence": row["sentence"],
                    "is_direct_mention": row.get("is_direct_mention", False),
                }
            )

        for (source_file, entity), gdata in grouped.items():
            if gdata["link_type"] is None:
                continue
            items.append(
                {
                    "sample_name": gdata["sample_name"],
                    "source_file": source_file,
                    "entity": entity,
                    "true_type": gdata["link_type"].lower(),
                    "sentences_meta": gdata["sentences_meta"],
                    "allowed_types": gdata["allowed_types"],
                }
            )

    return items


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    all_classes = sorted(set(y_true) | set(y_pred))
    true_classes = sorted(set(y_true))
    per_class: dict[str, dict] = {}
    for cls in all_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == cls),
        }

    total_support = len(y_true)
    if true_classes:
        macro_p = sum(per_class[c]["precision"] for c in true_classes) / len(
            true_classes
        )
        macro_r = sum(per_class[c]["recall"] for c in true_classes) / len(true_classes)
        macro_f1 = sum(per_class[c]["f1"] for c in true_classes) / len(true_classes)
        weighted_p = (
            sum(
                per_class[c]["precision"] * per_class[c]["support"]
                for c in true_classes
            )
            / total_support
        )
        weighted_r = (
            sum(per_class[c]["recall"] * per_class[c]["support"] for c in true_classes)
            / total_support
        )
        weighted_f1 = (
            sum(per_class[c]["f1"] * per_class[c]["support"] for c in true_classes)
            / total_support
        )
    else:
        macro_p = macro_r = macro_f1 = 0.0
        weighted_p = weighted_r = weighted_f1 = 0.0

    return {
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
    }


_W = 75
_ROW = "{:<25} | {:>10} | {:>8} | {:>8} | {:>8}"
_HDR = _ROW.format("Class", "Precision", "Recall", "F1", "Support")


def _print_metrics(metrics: dict, title: str = "OVERALL", n: int = 0) -> None:
    label = f"  {title}  [{n} pairs]" if n else f"  {title}"
    print(f"\n{'━' * _W}")
    print(label)
    print("━" * _W)
    print(_HDR)
    print("─" * _W)
    for cls, m in sorted(metrics["per_class"].items()):
        print(
            _ROW.format(
                cls,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                m["support"],
            )
        )
    print("─" * _W)
    macro = metrics["macro"]
    print(
        _ROW.format(
            "macro avg",
            f"{macro['precision']:.4f}",
            f"{macro['recall']:.4f}",
            f"{macro['f1']:.4f}",
            "",
        )
    )
    weighted = metrics["weighted"]
    print(
        _ROW.format(
            "weighted avg",
            f"{weighted['precision']:.4f}",
            f"{weighted['recall']:.4f}",
            f"{weighted['f1']:.4f}",
            "",
        )
    )


def _format_batch_item(
    idx: int, source_file: str, entity: str, sentences: list[str]
) -> str:
    source_clean = Path(source_file).stem.replace("_", " ")
    entity_clean = entity.replace("_", " ")
    sents_formatted = "\n".join(f"    {i + 1}. {s}" for i, s in enumerate(sentences))
    return (
        f'[{idx}] Source: "{source_clean}"  →  Entity: "{entity_clean}"\n'
        f"  Context sentences:\n{sents_formatted}"
    )


def _build_messages(items: list[dict], allowed_types: list[str]) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    relation_types_str = ", ".join(allowed_types)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(relation_types=relation_types_str)
    items_str = "\n\n".join(
        _format_batch_item(i, it["source_file"], it["entity"], it["sentences"])
        for i, it in enumerate(items)
    )
    human_content = HUMAN_PROMPT_TEMPLATE.format(count=len(items), items=items_str)

    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider == "google":
        return [
            HumanMessage(content=system_content),
            HumanMessage(content=human_content),
        ]
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


def _parse_llm_response(raw_text: str, n_items: int) -> list[Optional[dict]]:
    if not raw_text or not raw_text.strip():
        logger.warning("LLM returned an empty response")
        return [None] * n_items
    parsed = json_repair.loads(raw_text)
    if not isinstance(parsed, (dict, list)):
        logger.warning(f"LLM returned unparseable response: {raw_text[:100]!r}")
        return [None] * n_items
    if isinstance(parsed, list):
        result_dict = None
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "thinking":
                continue
            if "results" in item:
                result_dict = item
                break
        if result_dict is None:
            result_dict = {
                "results": [
                    i
                    for i in parsed
                    if isinstance(i, dict) and i.get("type") != "thinking"
                ]
            }
        parsed = result_dict
    batch_response = BatchResponse(**parsed)
    out: list[Optional[dict]] = [None] * n_items
    for pred in batch_response.results:
        if 0 <= pred.id < n_items:
            out[pred.id] = {
                "predicted_type": pred.predicted_type.lower(),
                "reasoning": pred.reasoning,
            }
    return out


def _extract_raw_text(response: Any) -> str:
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, list):
        return "".join(
            c.get("text") or c.get("content") or str(c)
            if isinstance(c, dict)
            else (c.text if hasattr(c, "text") else str(c))
            for c in content
        )
    return str(content)


def _invoke_llm_batch(
    llm: Any,
    items: list[dict],
    allowed_types: list[str],
) -> list[Optional[dict]]:
    messages = _build_messages(items, allowed_types)
    response = llm.invoke(messages)
    return _parse_llm_response(_extract_raw_text(response), len(items))


async def _ainvoke_llm_batch(
    llm: Any,
    items: list[dict],
    allowed_types: list[str],
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict], list[Optional[dict]]]:
    async with semaphore:
        messages = _build_messages(items, allowed_types)
        response = await llm.ainvoke(messages)
        return items, _parse_llm_response(_extract_raw_text(response), len(items))


async def _run_async_classification(
    llm: Any,
    to_process: list[dict],
    batch_size: int,
    concurrent_requests: int,
    output_path: Path,
    all_results: list[dict],
) -> None:
    batches = [
        to_process[s : s + batch_size] for s in range(0, len(to_process), batch_size)
    ]
    semaphore = asyncio.Semaphore(concurrent_requests)
    write_lock = asyncio.Lock()

    async def process_batch(batch: list[dict], out_f) -> None:
        allowed_types = batch[0]["allowed_types"]
        try:
            items, preds = await _ainvoke_llm_batch(
                llm, batch, allowed_types, semaphore
            )
        except Exception as e:
            logger.error(f"Batch failed (entity={batch[0]['entity']}...): {e}")
            items, preds = batch, [None] * len(batch)

        async with write_lock:
            for item, pred in zip(items, preds):
                if pred is None:
                    logger.warning(
                        f"No prediction for pair (sample={item['sample_name']}, "
                        f"file={item['source_file']}, entity={item['entity']})"
                        " — will retry on restart"
                    )
                    continue
                result = {
                    "sample_name": item["sample_name"],
                    "source_file": item["source_file"],
                    "entity": item["entity"],
                    "true_type": item["true_type"],
                    "predicted_type": pred["predicted_type"],
                    "reasoning": pred["reasoning"],
                }
                all_results.append(result)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    with output_path.open("a", encoding="utf-8") as out_f:
        tasks = [asyncio.create_task(process_batch(b, out_f)) for b in batches]
        with tqdm(total=len(batches), desc="Batches", unit="batch") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM relation-classification accuracy against a gold dataset."
    )
    parser.add_argument(
        "--gold-dir",
        required=True,
        help="Directory containing gold JSON files (one per sample).",
    )
    parser.add_argument(
        "--vault-dir",
        required=True,
        help="Directory with vault sample subdirectories (each containing .kg_builder/config.json).",
    )
    parser.add_argument(
        "--output",
        default="results/classification_results.jsonl",
        help="Path to the output JSONL file (progress is saved after every batch).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top-ranked sentences to keep per pair (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of pairs per LLM prompt (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=DEFAULT_CONCURRENT_REQUESTS,
        help=f"Number of LLM requests to run simultaneously (default: {DEFAULT_CONCURRENT_REQUESTS}).",
    )
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    vault_dir = Path(args.vault_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = _load_gold(gold_dir, vault_dir)
    logger.info(f"Extracted {len(all_items)} ground truth relations")

    logger.info(f"Loading cross-encoder: {RERANKER_MODEL_NAME}")
    cross_encoder = CrossEncoder(
        RERANKER_MODEL_NAME,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    logger.info(f"Ranking contexts for {len(all_items)} pairs")
    ranked_items: list[dict] = []
    for item in tqdm(all_items, desc="Ranking contexts"):
        source_clean = Path(item["source_file"]).stem.replace("_", " ")
        entity_clean = item["entity"].replace("_", " ")
        query = f"What is the relationship between {source_clean} and {entity_clean}?"
        top_sentences = _rank_sentences(
            cross_encoder, query, item["sentences_meta"], args.top_n
        )
        ranked_items.append(
            {
                "sample_name": item["sample_name"],
                "source_file": item["source_file"],
                "entity": item["entity"],
                "true_type": item["true_type"],
                "sentences": top_sentences,
                "allowed_types": item["allowed_types"],
            }
        )

    processed_keys: set[tuple] = set()
    all_results: list[dict] = []
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                all_results.append(r)
                processed_keys.add((r["sample_name"], r["source_file"], r["entity"]))
        if all_results:
            logger.info(
                f"Loaded {len(all_results)} previously completed pairs from checkpoint"
            )

    to_process = [
        it
        for it in ranked_items
        if (it["sample_name"], it["source_file"], it["entity"]) not in processed_keys
    ]

    if not to_process:
        logger.info(
            "All pairs already processed — printing metrics and clearing checkpoint."
        )
    else:
        logger.info(f"Pairs remaining: {len(to_process)}")
        llm = _load_llm()

        try:
            asyncio.run(
                _run_async_classification(
                    llm=llm,
                    to_process=to_process,
                    batch_size=args.batch_size,
                    concurrent_requests=args.concurrent_requests,
                    output_path=output_path,
                    all_results=all_results,
                )
            )
        except KeyboardInterrupt:
            tqdm.write("\nInterrupted — computing metrics on results saved so far.")

    valid = [r for r in all_results if r.get("predicted_type") is not None]
    if not valid:
        logger.warning("No valid predictions to evaluate.")
        return

    logger.info(f"Computing metrics for {len(valid)} predictions")

    by_sample_results: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_sample_results[r["sample_name"]].append(r)

    for sample_name, sample_results in sorted(by_sample_results.items()):
        y_true_s = [r["true_type"] for r in sample_results]
        y_pred_s = [r["predicted_type"] for r in sample_results]
        _print_metrics(
            _compute_metrics(y_true_s, y_pred_s),
            title=f"Sample: {sample_name}",
            n=len(sample_results),
        )

    y_true = [r["true_type"] for r in valid]
    y_pred = [r["predicted_type"] for r in valid]
    _print_metrics(_compute_metrics(y_true, y_pred), title="OVERALL", n=len(valid))
    print()

    if not to_process and output_path.exists():
        output_path.unlink()
        logger.info(f"Checkpoint cleared: {output_path}")


if __name__ == "__main__":
    main()
