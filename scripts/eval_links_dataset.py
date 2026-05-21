from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

LINK_PATTERN = re.compile(
    r"^\s*(?:[-*+]\s+)?(?P<relation>.+?)\s*::\s*\[\[(?P<target>.+?)\]\]\s*$",
    re.MULTILINE,
)

DEFAULT_HEADER_LINE = "## Related Connections"


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


@dataclass(frozen=True)
class TypedLink:
    source: str
    relation: str
    target: str


@dataclass(frozen=True)
class UntypedLink:
    source: str
    target: str


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_header_line_from_marker(marker: str) -> str:
    for line in marker.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return DEFAULT_HEADER_LINE


def _split_connections_block(content: str, header_marker: Optional[str]) -> str:
    if not content:
        return ""

    if header_marker:
        marker = header_marker
        if marker and marker in content:
            return content.split(marker, 1)[1]

        header_line = _extract_header_line_from_marker(marker)
    else:
        header_line = DEFAULT_HEADER_LINE

    header_re = re.compile(rf"^\s*{re.escape(header_line)}\s*$", re.MULTILINE)
    match = header_re.search(content)
    if not match:
        return ""

    return content[match.end() :]


def _normalize_target(raw: str) -> str:
    target = raw.strip()
    target = target.split("|", 1)[0].strip()
    target = target.split("#", 1)[0].strip()
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target


def _normalize_relation(raw: str) -> str:
    return " ".join(raw.strip().split()).casefold()


def extract_links_from_markdown(
    md_path: Path, *, header_marker: Optional[str]
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    content = _safe_read_text(md_path)
    block = _split_connections_block(content, header_marker)
    if not block:
        return set(), set()

    typed: Set[Tuple[str, str]] = set()
    untyped: Set[str] = set()

    for match in LINK_PATTERN.finditer(block):
        relation = _normalize_relation(match.group("relation"))
        target = _normalize_target(match.group("target"))
        if not relation or not target:
            continue
        typed.add((relation, target))
        untyped.add(target)

    return typed, untyped


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.md"):
        parts = set(path.parts)
        if ".obsidian" in parts or ".kg_builder" in parts:
            continue
        yield path


def _load_sample_header_marker(sample_dir: Path) -> Optional[str]:
    cfg_path = sample_dir / ".kg_builder" / "config.json"
    if not cfg_path.exists():
        return None

    try:
        cfg = json.loads(_safe_read_text(cfg_path))
    except Exception:
        return None

    marker = cfg.get("link_header")
    if isinstance(marker, str) and marker.strip():
        return marker
    return None


def build_link_sets(
    sample_dir: Path,
) -> Tuple[Set[TypedLink], Set[UntypedLink], Dict[str, Set[Tuple[str, str]]]]:
    header_marker = _load_sample_header_marker(sample_dir)

    typed_links: Set[TypedLink] = set()
    untyped_links: Set[UntypedLink] = set()
    typed_by_relation: Dict[str, Set[Tuple[str, str]]] = {}

    for md_path in _iter_markdown_files(sample_dir):
        rel_source = md_path.relative_to(sample_dir).as_posix()
        typed_pairs, untyped_targets = extract_links_from_markdown(
            md_path, header_marker=header_marker
        )

        for relation, target in typed_pairs:
            typed_links.add(TypedLink(rel_source, relation, target))
            typed_by_relation.setdefault(relation, set()).add((rel_source, target))

        for target in untyped_targets:
            untyped_links.add(UntypedLink(rel_source, target))

    return typed_links, untyped_links, typed_by_relation


def counts_from_sets(gt: Set[object], pred: Set[object]) -> Counts:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return Counts(tp=tp, fp=fp, fn=fn)


def weighted_average(
    metrics: Mapping[str, Counts], supports: Mapping[str, int]
) -> Dict[str, float]:
    total_support = sum(supports.values())
    if total_support <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    wp = sum(metrics[k].precision * supports.get(k, 0) for k in metrics) / total_support
    wr = sum(metrics[k].recall * supports.get(k, 0) for k in metrics) / total_support
    wf1 = sum(metrics[k].f1 * supports.get(k, 0) for k in metrics) / total_support
    return {"precision": wp, "recall": wr, "f1": wf1}


def evaluate_sample(gt_sample: Path, pred_sample: Path) -> Dict[str, object]:
    gt_typed, gt_untyped, gt_by_rel = build_link_sets(gt_sample)
    pred_typed, pred_untyped, pred_by_rel = build_link_sets(pred_sample)

    typed_micro = counts_from_sets(gt_typed, pred_typed)
    untyped_micro = counts_from_sets(gt_untyped, pred_untyped)

    all_relations = sorted(set(gt_by_rel.keys()) | set(pred_by_rel.keys()))
    per_rel: Dict[str, Dict[str, object]] = {}
    per_rel_counts: Dict[str, Counts] = {}
    supports: Dict[str, int] = {}

    for rel in all_relations:
        gt_set = gt_by_rel.get(rel, set())
        pred_set = pred_by_rel.get(rel, set())
        c = counts_from_sets(gt_set, pred_set)
        per_rel_counts[rel] = c
        supports[rel] = len(gt_set)
        per_rel[rel] = {
            "tp": c.tp,
            "fp": c.fp,
            "fn": c.fn,
            "precision": c.precision,
            "recall": c.recall,
            "f1": c.f1,
            "support": len(gt_set),
        }

    typed_weighted = weighted_average(per_rel_counts, supports)

    return {
        "typed_micro": {
            "tp": typed_micro.tp,
            "fp": typed_micro.fp,
            "fn": typed_micro.fn,
            "precision": typed_micro.precision,
            "recall": typed_micro.recall,
            "f1": typed_micro.f1,
        },
        "untyped_micro": {
            "tp": untyped_micro.tp,
            "fp": untyped_micro.fp,
            "fn": untyped_micro.fn,
            "precision": untyped_micro.precision,
            "recall": untyped_micro.recall,
            "f1": untyped_micro.f1,
        },
        "typed_weighted_by_relation": typed_weighted,
        "per_relation": per_rel,
        "counts": {
            "gt_typed": len(gt_typed),
            "pred_typed": len(pred_typed),
            "gt_untyped": len(gt_untyped),
            "pred_untyped": len(pred_untyped),
        },
    }


def _list_sample_dirs(root: Path) -> List[str]:
    if not root.is_dir():
        return []

    names: List[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name in {".obsidian"}:
            continue

        has_md = any(child.glob("*.md"))
        has_meta = (child / ".meta").exists()
        if has_md or has_meta:
            names.append(child.name)
    return sorted(names)


def _format_counts(label: str, c: Mapping[str, float | int]) -> str:
    return (
        f"{label}: P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.4f} "
        f"(TP={c['tp']} FP={c['fp']} FN={c['fn']})"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted KG links for multiple dataset samples",
    )
    parser.add_argument(
        "--gold-root",
        required=True,
        type=Path,
        help="Root directory containing ground-truth sample vaults (e.g. data/test_vaults/gold)",
    )
    parser.add_argument(
        "--pred-root",
        required=True,
        type=Path,
        help="Root directory containing predicted sample vaults (e.g. results/links)",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional list of sample folder names to evaluate; default: intersection of gold/pred",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    gold_root: Path = args.gold_root
    pred_root: Path = args.pred_root

    if not gold_root.is_dir():
        raise SystemExit(f"gold root is not a directory: {gold_root}")
    if not pred_root.is_dir():
        raise SystemExit(f"pred root is not a directory: {pred_root}")

    if args.samples:
        sample_names = list(args.samples)
    else:
        gold_samples = set(_list_sample_dirs(gold_root))
        pred_samples = set(_list_sample_dirs(pred_root))
        sample_names = sorted(gold_samples & pred_samples)

    if not sample_names:
        raise SystemExit("No samples found to evaluate (check roots and folder names).")

    report: Dict[str, object] = {
        "gold_root": str(gold_root),
        "pred_root": str(pred_root),
        "samples": {},
    }

    overall_typed = Counts()
    overall_untyped = Counts()

    print("=" * 60)
    print("LINK EVALUATION (typed + untyped)")
    print(f"gold: {gold_root}")
    print(f"pred: {pred_root}")
    print("=" * 60)

    for sample in sample_names:
        gt_sample = gold_root / sample
        pred_sample = pred_root / sample
        if not gt_sample.is_dir() or not pred_sample.is_dir():
            continue

        sample_result = evaluate_sample(gt_sample, pred_sample)
        report["samples"][sample] = sample_result

        typed_micro = sample_result["typed_micro"]
        untyped_micro = sample_result["untyped_micro"]

        overall_typed.add(
            Counts(
                tp=int(typed_micro["tp"]),
                fp=int(typed_micro["fp"]),
                fn=int(typed_micro["fn"]),
            )
        )
        overall_untyped.add(
            Counts(
                tp=int(untyped_micro["tp"]),
                fp=int(untyped_micro["fp"]),
                fn=int(untyped_micro["fn"]),
            )
        )

        print(f"\n[{sample}]")
        print(_format_counts(" typed   ", typed_micro))
        print(_format_counts(" untyped ", untyped_micro))
        w = sample_result["typed_weighted_by_relation"]
        print(
            f" weighted(typed by relation): P={w['precision']:.4f} R={w['recall']:.4f} F1={w['f1']:.4f}"
        )

    overall = {
        "typed_micro": {
            "tp": overall_typed.tp,
            "fp": overall_typed.fp,
            "fn": overall_typed.fn,
            "precision": overall_typed.precision,
            "recall": overall_typed.recall,
            "f1": overall_typed.f1,
        },
        "untyped_micro": {
            "tp": overall_untyped.tp,
            "fp": overall_untyped.fp,
            "fn": overall_untyped.fn,
            "precision": overall_untyped.precision,
            "recall": overall_untyped.recall,
            "f1": overall_untyped.f1,
        },
    }

    report["overall"] = overall

    print("\n" + "=" * 60)
    print("OVERALL")
    print(_format_counts(" typed   ", overall["typed_micro"]))
    print(_format_counts(" untyped ", overall["untyped_micro"]))
    print("=" * 60)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote JSON report to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
