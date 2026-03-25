import argparse
import logging
import re
from pathlib import Path
from typing import Set, Tuple, Union

from src.kg_builder.config import DEFAULT_LOG_LEVEL, LINK_HEADER, setup_logging

logger = logging.getLogger(__name__)


LINK_PATTERN = re.compile(
    r"^\s*(?:[-*+]\s+)?(?P<relation>.*?)\s*::\s*\[\[(?P<target>.*?)\]\]", re.MULTILINE
)


class KGMetrics:
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
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def extract_links_from_file(
    file_path: Path, ignore_class: bool = False
) -> Set[Union[str, Tuple[str, str]]]:
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Could not read file {file_path}: {e}")
        return set()

    header_marker = LINK_HEADER.strip()
    if header_marker not in content:
        return set()

    _, connections_block = content.split(header_marker, 1)

    links = set()
    for match in LINK_PATTERN.finditer(connections_block):
        relation = match.group("relation").strip()
        target = match.group("target").strip()

        if ignore_class:
            links.add(target)
        else:
            links.add((relation, target))

    return links


def evaluate_vaults(gt_path: Path, pred_path: Path, ignore_class: bool = False):
    metrics = KGMetrics()

    gt_files = {p.relative_to(gt_path): p for p in gt_path.rglob("*.md")}
    pred_files = {p.relative_to(pred_path): p for p in pred_path.rglob("*.md")}

    all_rel_paths = set(gt_files.keys()) | set(pred_files.keys())

    for rel_path in sorted(all_rel_paths):
        gt_file = gt_files.get(rel_path)
        pred_file = pred_files.get(rel_path)

        gt_links = extract_links_from_file(gt_file, ignore_class) if gt_file else set()
        pred_links = (
            extract_links_from_file(pred_file, ignore_class) if pred_file else set()
        )

        tp_set = gt_links & pred_links
        fp_set = pred_links - gt_links
        fn_set = gt_links - pred_links

        metrics.tp += len(tp_set)
        metrics.fp += len(fp_set)
        metrics.fn += len(fn_set)

        if len(fp_set) > 0 or len(fn_set) > 0:
            logger.debug(
                f"File: {rel_path} | TP: {len(tp_set)}, FP: {len(fp_set)}, FN: {len(fn_set)}"
            )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="KG Evaluator: Precision, Recall and F1-score for Knowledge Graph links"
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Path to Ground Truth vault"
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Path to Predicted vault"
    )
    parser.add_argument(
        "--ignore-class",
        action="store_true",
        help="Ignore relation type, only check if link to target exists",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    gt_path = Path(args.gt).resolve()
    pred_path = Path(args.pred).resolve()

    if not gt_path.is_dir():
        print(f"Error: GT path '{gt_path}' is not a directory.")
        return
    if not pred_path.is_dir():
        print(f"Error: Pred path '{pred_path}' is not a directory.")
        return

    setup_logging(Path("."), log_level=args.log_level)

    logger.info("Starting evaluation...")
    logger.info(f"GT Vault: {gt_path}")
    logger.info(f"Pred Vault: {pred_path}")
    logger.info(f"Ignore relation type: {args.ignore_class}")

    metrics = evaluate_vaults(gt_path, pred_path, args.ignore_class)

    print("" + "=" * 30)
    print("      EVALUATION RESULTS")
    print("=" * 30)
    print(f"True Positives:  {metrics.tp}")
    print(f"False Positives: {metrics.fp}")
    print(f"False Negatives: {metrics.fn}")
    print("-" * 30)
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1-score:  {metrics.f1:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
