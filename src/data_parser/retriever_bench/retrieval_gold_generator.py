import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.kg_builder.config import META_DIR_NAME, OUTPUT_LINKS_FILE_NAME
from src.kg_builder.models import DocumentEntity
from src.kg_builder.retrieval import StrictRetrievalStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_sample_lang(sample_dir: Path, default_lang: str) -> str:
    config_path = sample_dir / META_DIR_NAME / OUTPUT_LINKS_FILE_NAME
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if lang := data.get("lang"):
                return lang
        except (json.JSONDecodeError, IOError):
            pass
    return default_lang


class MockVectorStore:
    def __init__(self, lang: str):
        self.lang = lang


def get_entities_from_dir(directory: Path) -> Dict[str, DocumentEntity]:
    entities = {}
    for md_file in directory.glob("*.md"):
        rel_path = str(md_file)
        title = md_file.stem
        entities[rel_path] = DocumentEntity(rel_path=rel_path, title=title, aliases=[])
    return entities


def process_sample(sample_dir: Path, lang: str) -> List[Dict[str, Any]]:
    """Processes a single sample directory and returns findings."""
    logger.info(f"Processing sample: {sample_dir.name}")

    entities = get_entities_from_dir(sample_dir)
    if not entities:
        logger.warning(f"No .md files found in {sample_dir}")
        return []

    mock_vs = MockVectorStore(lang=lang)
    strategy = StrictRetrievalStrategy(
        vector_store=mock_vs,
        global_entity_dict=entities,
        context_sents_before=0,
        context_sents_after=0,
    )

    results = []

    full_docs = {}
    for file_path_str in entities.keys():
        file_path = Path(file_path_str)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_docs[file_path_str] = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")

    _, initial_meta = strategy.retrieve(chunks=[], full_docs=full_docs)

    for meta in initial_meta:
        results.append(
            {
                "entity": Path(meta["target_path"]).stem,
                "source_file": Path(meta["source_path"]).name,
                "sentence": meta["target_content"],
            }
        )

    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully saved {len(results)} findings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate retrieval validation dataset from gold vaults."
    )
    parser.add_argument(
        "--dir",
        "--vault",
        dest="vault_dir",
        required=True,
        help="Path to the 'gold' directory containing samples.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output directory or base filename.",
    )
    parser.add_argument(
        "--lang",
        default="ru",
        choices=["ru", "en"],
        help="Language of the articles (default: ru).",
    )

    args = parser.parse_args()

    vault_path = Path(args.vault_dir)
    if not vault_path.is_dir():
        logger.error(f"Directory {vault_path} does not exist.")
        return

    output_base = Path(args.output)
    if output_base.suffix == ".json":
        output_dir = output_base.parent
    else:
        output_dir = output_base

    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = [
        d for d in vault_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    if not subdirs:
        lang = get_sample_lang(vault_path, args.lang)
        results = process_sample(vault_path, lang)
        save_results(results, output_dir / f"{vault_path.name}.json")
    else:
        for subdir in subdirs:
            lang = get_sample_lang(subdir, args.lang)
            results = process_sample(subdir, lang)
            save_results(results, output_dir / f"{subdir.name}.json")


if __name__ == "__main__":
    main()
