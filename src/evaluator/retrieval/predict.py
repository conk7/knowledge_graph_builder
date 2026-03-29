import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from src.kg_builder.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    INITIAL_RETRIEVAL_K,
    LINK_HEADER,
    LINKS_CONFIG_FILE_NAME,
    META_DIR_NAME,
    RERANKER_MODEL_NAME,
    RERANKER_THRESHOLD,
    RERANKER_TOP_K,
    VECTOR_SEARCH_WEIGHT,
)
from src.kg_builder.models import DocumentEntity, NewlyAddedChunk
from src.kg_builder.retrieval import (
    CombinedRetrievalStrategy,
    StrictRetrievalStrategy,
    VectorSearchRerankStrategy,
)
from src.kg_builder.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_sample_lang(sample_dir: Path) -> str:
    config_path = sample_dir / META_DIR_NAME / LINKS_CONFIG_FILE_NAME
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("lang") or "en"
        except (json.JSONDecodeError, IOError):
            pass
    return "en"


def get_entities_from_dir(directory: Path) -> Dict[str, DocumentEntity]:
    return {
        str(md_file): DocumentEntity(
            rel_path=str(md_file), title=md_file.stem, aliases=[]
        )
        for md_file in directory.glob("*.md")
    }


def _cands_to_dicts(candidates) -> List[Dict[str, Any]]:
    return [
        {
            "entity": cand.target_path.stem,
            "source_file": cand.source_path.name,
            "sentence": cand.target_content,
        }
        for cand in candidates
    ]


def predict_sample(
    sample_dir: Path,
    lang: str,
    show_progress: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    entities = get_entities_from_dir(sample_dir)
    if not entities:
        logger.warning(f"No .md files in {sample_dir}")
        return {}

    full_docs: Dict[str, str] = {}
    for fp_str in entities:
        fp = Path(fp_str)
        try:
            content = fp.read_text(encoding="utf-8")
            if LINK_HEADER in content:
                content = content[: content.index(LINK_HEADER)]
            full_docs[fp_str] = content
        except Exception as e:
            logger.error(f"Failed to read {fp}: {e}")

    temp_dir = tempfile.mkdtemp()
    try:
        vs = VectorStore(
            index_path=Path(temp_dir),
            embedding_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            splitter_type="sentence_window",
            separators=CHUNK_SEPARATORS,
            vector_weight=VECTOR_SEARCH_WEIGHT,
            fresh_start=True,
            lang=lang,
        )

        all_chunks: List[NewlyAddedChunk] = []
        for fp, content in full_docs.items():
            for chunk_text in vs.add_document(fp, content):
                all_chunks.append(NewlyAddedChunk(file_path=fp, content=chunk_text))

        vs.rebuild_fts_index()

        strict_strat = StrictRetrievalStrategy(
            vector_store=vs,
            global_entity_dict=entities,
            context_sents_before=0,
            context_sents_after=0,
        )
        broad_strat = VectorSearchRerankStrategy(
            vector_store=vs,
            retrieval_k=INITIAL_RETRIEVAL_K,
            reranker_top_k=RERANKER_TOP_K,
            reranker_threshold=RERANKER_THRESHOLD,
        )
        combined_strat = CombinedRetrievalStrategy(
            strict_strat=strict_strat,
            broad_strat=broad_strat,
        )

        vs.load_reranker()
        try:
            strict_cands, _ = strict_strat.retrieve(
                all_chunks, full_docs, show_progress=show_progress
            )
            broad_cands, _ = broad_strat.retrieve(
                all_chunks, full_docs, show_progress=show_progress
            )
            combined_cands, _ = combined_strat.retrieve(
                all_chunks, full_docs, show_progress=show_progress
            )
        finally:
            vs.unload_reranker()

        return {
            "Strict": _cands_to_dicts(strict_cands),
            "Broad": _cands_to_dicts(broad_cands),
            "Combined": _cands_to_dicts(combined_cands),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval strategies on a vault and save prediction JSONs."
    )
    parser.add_argument(
        "--vault",
        required=True,
        help="Path to vault directory containing sample subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory; predictions saved as Strategy/sample.json.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show inner tqdm progress bars during retrieval.",
    )
    args = parser.parse_args()

    vault_path = Path(args.vault)
    if not vault_path.is_dir():
        logger.error(f"Vault directory not found: {vault_path}")
        return

    output_dir = Path(args.output_dir)
    for name in ("Strict", "Broad", "Combined"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    subdirs = [
        d for d in vault_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    if not subdirs:
        subdirs = [vault_path]

    for subdir in sorted(subdirs):
        logger.info(f"Predicting: {subdir.name}")
        lang = get_sample_lang(subdir)
        preds = predict_sample(subdir, lang, show_progress=args.show_progress)
        for strategy, items in preds.items():
            out_file = output_dir / strategy / f"{subdir.name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            logger.info(f"  [{strategy}] {len(items)} items → {out_file}")


if __name__ == "__main__":
    main()
