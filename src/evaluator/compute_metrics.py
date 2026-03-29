import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FUZZY_MATCH_THRESHOLD = 0.5

_nlp_cache: Dict[str, Any] = {}


class Metrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _get_nlp(lang: str):
    if lang not in _nlp_cache:
        import spacy

        model = "ru_core_news_sm" if lang == "ru" else "en_core_web_sm"
        _nlp_cache[lang] = spacy.load(model, disable=["parser", "ner"])
    return _nlp_cache[lang]


def _lemma_set(text: str, nlp) -> set:
    return {
        token.lemma_.lower()
        for token in nlp(text)
        if not token.is_stop and not token.is_punct and token.is_alpha
    }


def is_fuzzy_match(
    gold_sent: str,
    pred_sent: str,
    nlp,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> bool:
    gold_lemmas = _lemma_set(gold_sent, nlp)
    pred_lemmas = _lemma_set(pred_sent, nlp)
    if not gold_lemmas:
        return gold_sent.strip() in pred_sent
    return len(gold_lemmas & pred_lemmas) / len(gold_lemmas) >= threshold


def compute_metrics(
    gold_data: List[Dict[str, Any]],
    pred_data: List[Dict[str, Any]],
    nlp,
) -> Metrics:
    metrics = Metrics()

    pred_map: Dict[tuple, List[str]] = {}
    for item in pred_data:
        key = (item["source_file"], item["entity"])
        pred_map.setdefault(key, []).append(item["sentence"])

    for gold_item in gold_data:
        key = (gold_item["source_file"], gold_item["entity"])
        gold_sent = gold_item["sentence"].strip()
        if key in pred_map:
            if any(is_fuzzy_match(gold_sent, s, nlp) for s in pred_map[key]):
                metrics.tp += 1
            else:
                metrics.fn += 1
        else:
            metrics.fn += 1

    for key, sentences in pred_map.items():
        matched = any(
            key == (g["source_file"], g["entity"])
            and is_fuzzy_match(g["sentence"].strip(), s, nlp)
            for g in gold_data
            for s in sentences
        )
        if not matched:
            metrics.fp += 1

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute retrieval metrics from prediction and gold JSON files."
    )
    parser.add_argument(
        "--gold-dir",
        required=True,
        help="Directory containing gold JSON files (one per sample).",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory with per-strategy subdirs, each containing sample JSONs.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="spaCy language for fuzzy matching: 'en' or 'ru' (default: en).",
    )
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    pred_dir = Path(args.pred_dir)

    if not gold_dir.is_dir():
        logger.error(f"Gold dir not found: {gold_dir}")
        return
    if not pred_dir.is_dir():
        logger.error(f"Pred dir not found: {pred_dir}")
        return

    strategy_dirs = sorted(
        d for d in pred_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not strategy_dirs:
        logger.error(f"No strategy subdirs found in {pred_dir}")
        return

    nlp = _get_nlp(args.lang)

    total: Dict[str, Metrics] = {}
    for strat_dir in strategy_dirs:
        name = strat_dir.name
        agg = Metrics()

        pred_files = sorted(strat_dir.glob("*.json"))
        if not pred_files:
            logger.warning(f"No prediction JSONs in {strat_dir}")
            continue

        for pred_file in pred_files:
            gold_file = gold_dir / pred_file.name
            if not gold_file.exists():
                logger.warning(f"No gold file for {pred_file.name}, skipping.")
                continue

            with open(gold_file, encoding="utf-8") as f:
                gold_data = json.load(f)
            with open(pred_file, encoding="utf-8") as f:
                pred_data = json.load(f)

            m = compute_metrics(gold_data, pred_data, nlp)
            agg.tp += m.tp
            agg.fp += m.fp
            agg.fn += m.fn

        total[name] = agg

    print("\n" + "=" * 65)
    print(
        f"{'Strategy':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1':<10}"
    )
    print("-" * 65)
    for name, m in total.items():
        print(
            f"{name:<15} | {m.tp:<5} | {m.fp:<5} | {m.fn:<5} | "
            f"{m.precision:10.4f} | {m.recall:10.4f} | {m.f1:10.4f}"
        )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
