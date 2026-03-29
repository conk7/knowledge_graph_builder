import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from .compute_metrics import Metrics, _get_nlp, compute_metrics
from .predict import get_sample_lang, predict_sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval strategies and evaluate against gold JSONs."
    )
    parser.add_argument(
        "--vault",
        required=True,
        help="Path to vault directory containing sample subdirectories.",
    )
    parser.add_argument(
        "--gold-dir",
        required=True,
        help="Directory containing gold JSON files (one per sample).",
    )
    parser.add_argument(
        "--output-dir",
        help="If given, prediction JSONs are saved as Strategy/sample.json.",
    )
    parser.add_argument(
        "--show-inner-progress",
        action="store_true",
        help="Show nested tqdm progress bars during retrieval.",
    )
    args = parser.parse_args()

    vault_path = Path(args.vault)
    gold_dir = Path(args.gold_dir)

    if not vault_path.is_dir():
        logger.error(f"Vault directory not found: {vault_path}")
        return
    if not gold_dir.is_dir():
        logger.error(f"Gold directory not found: {gold_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        for name in ("Strict", "Broad", "Combined"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)

    subdirs = [
        d for d in vault_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    if not subdirs:
        subdirs = [vault_path]

    total_metrics: Dict[str, Metrics] = {
        "Strict": Metrics(),
        "Broad": Metrics(),
        "Combined": Metrics(),
    }

    for subdir in sorted(subdirs):
        gold_file = gold_dir / f"{subdir.name}.json"
        if not gold_file.exists():
            logger.warning(f"No gold file for {subdir.name}, skipping.")
            continue

        with open(gold_file, encoding="utf-8") as f:
            gold_data = json.load(f)

        lang = get_sample_lang(subdir)
        logger.info(f"Processing: {subdir.name} (lang={lang})")

        preds = predict_sample(subdir, lang, show_progress=args.show_inner_progress)
        if not preds:
            continue

        if output_dir:
            for strategy, items in preds.items():
                out_file = output_dir / strategy / f"{subdir.name}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)

        nlp = _get_nlp(lang)
        for strategy, pred_data in preds.items():
            m = compute_metrics(gold_data, pred_data, nlp)
            total_metrics[strategy].tp += m.tp
            total_metrics[strategy].fp += m.fp
            total_metrics[strategy].fn += m.fn

    print("\n" + "=" * 65)
    print(
        f"{'Strategy':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1':<10}"
    )
    print("-" * 65)
    for name, m in total_metrics.items():
        print(
            f"{name:<15} | {m.tp:<5} | {m.fp:<5} | {m.fn:<5} | "
            f"{m.precision:10.4f} | {m.recall:10.4f} | {m.f1:10.4f}"
        )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
