import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import json_repair
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.kg_builder.config import DEFAULT_LINK_TYPES

logger = logging.getLogger(__name__)


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 5
DEFAULT_BATCH_SIZE = 10


class RelationPrediction(BaseModel):
    id: int = Field(description="0-based index of the item within the current batch")
    reasoning: str = Field(description="One-sentence explanation")
    predicted_type: str = Field(
        description="Exactly one allowed relation type, or 'no link'"
    )


class BatchResponse(BaseModel):
    results: list[RelationPrediction]


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Ontology Engineer tasked with classifying semantic relationships \
between pairs of knowledge-graph notes.

For each item you will receive:
  - A source document name and a target entity name.
  - A ranked list of context sentences extracted from the source document that \
mention the target entity.

Your job: decide which single relationship best describes how the source document \
relates to the target entity.

Allowed relation types (use EXACTLY one, case-insensitive): {relation_types}
If no clear relationship exists, output "no link".

Respond with ONLY a valid JSON object — no markdown fences, no extra text — \
matching this structure:
  "results": list of objects, one per input item, each containing:
    "id" - integer, the item number from the input (0-based)
    "reasoning" - one sentence explaining the choice
    "predicted_type" - the chosen relation type or "no link"
"""

HUMAN_PROMPT_TEMPLATE = """\
Classify the following {count} pair(s):

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
        )

    elif provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        api_key = os.environ.get("CEREBRAS_API_KEY")
        llm = ChatCerebras(
            model=model or "llama3.1-8b",
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
        )
        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception:
            pass
        return llm

    elif provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        return ChatGroq(model=model, api_key=api_key, temperature=temperature)

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
    classes = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict] = {}
    for cls in classes:
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

    if per_class:
        macro_p = sum(v["precision"] for v in per_class.values()) / len(per_class)
        macro_r = sum(v["recall"] for v in per_class.values()) / len(per_class)
        macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    return {
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
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


def _invoke_llm_batch(
    llm: Any,
    items: list[dict],
    allowed_types: list[str],
) -> list[Optional[dict]]:
    from langchain_core.messages import HumanMessage, SystemMessage

    relation_types_str = ", ".join(allowed_types)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(relation_types=relation_types_str)
    items_str = "\n\n".join(
        _format_batch_item(i, it["source_file"], it["entity"], it["sentences"])
        for i, it in enumerate(items)
    )
    human_content = HUMAN_PROMPT_TEMPLATE.format(count=len(items), items=items_str)

    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "google":
        messages = [
            HumanMessage(content=system_content),
            HumanMessage(content=human_content),
        ]
    else:
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    response = llm.invoke(messages)
    raw_text: str = response.content if hasattr(response, "content") else str(response)

    parsed = json_repair.loads(raw_text)
    batch_response = BatchResponse(**parsed)

    out: list[Optional[dict]] = [None] * len(items)
    for pred in batch_response.results:
        if 0 <= pred.id < len(items):
            out[pred.id] = {
                "predicted_type": pred.predicted_type.lower(),
                "reasoning": pred.reasoning,
            }
    return out


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
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    vault_dir = Path(args.vault_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = _load_gold(gold_dir, vault_dir)
    logger.info(f"Extracted {len(all_items)} ground truth relations")

    logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

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

        by_sample: dict[str, list[dict]] = defaultdict(list)
        for it in to_process:
            by_sample[it["sample_name"]].append(it)

        try:
            with output_path.open("a", encoding="utf-8") as out_f:
                for sample_name, sample_items in tqdm(
                    by_sample.items(), desc="Samples", unit="sample"
                ):
                    for batch_start in tqdm(
                        range(0, len(sample_items), args.batch_size),
                        desc=f"  Batches [{sample_name}]",
                        unit="batch",
                        leave=False,
                    ):
                        batch = sample_items[
                            batch_start : batch_start + args.batch_size
                        ]
                        allowed_types = batch[0]["allowed_types"]

                        try:
                            preds = _invoke_llm_batch(llm, batch, allowed_types)
                        except Exception as e:
                            logger.error(
                                f"Batch failed (sample={sample_name}, "
                                f"start={batch_start}): {e}"
                            )
                            preds = [None] * len(batch)

                        for item, pred in zip(batch, preds):
                            if pred is None:
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
