import json
import logging
from pathlib import Path
from typing import Dict

from src.kg_builder.config import DEFAULT_LINK_EN2RU_TRANSLATION

logger = logging.getLogger(__name__)


RELATION_MAP_RU: Dict[str, str] = {
    "instance of": "экземпляр",
    "subclass of": "подкласс",
    "part of": "часть",
    "has part": "содержит",
    "facet of": "аспект",
    "different from": "отличается от",
    "opposite of": "противоположность",
    "said to be the same as": "аналог",
    "studied in": "изучается в",
    "use": "использует",
    "field of work": "сфера деятельности",
    "main subject": "основная тема",
    "has cause": "имеет причину",
    "has effect": "имеет следствие",
}
RELATION_MAP_RU.update(
    {k.lower(): v for k, v in DEFAULT_LINK_EN2RU_TRANSLATION.items()}
)


def translate_relation(relation: str, lang: str) -> str:
    if not relation:
        return "Относится к" if lang == "ru" else "Related to"

    primary_relation = relation.split(",")[0].strip()
    rel_lower = primary_relation.lower()

    if lang == "ru":
        return RELATION_MAP_RU.get(rel_lower, primary_relation)

    return primary_relation


def apply_classification(
    vault_path: Path,
    classified_json: Path,
    links_header: str = "## Related Connections",
    lang: str = "ru",
) -> None:
    if not classified_json.exists():
        logger.error(f"Classified JSON not found: {classified_json}")
        return

    with open(classified_json, "r", encoding="utf-8") as f:
        classified_data = json.load(f)

    applied_count = 0
    for file_path in vault_path.glob("*.md"):
        file_name = file_path.stem
        links = classified_data.get(file_name, [])

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        parts = content.split(links_header)
        if len(parts) < 2:
            continue

        main_text = parts[0].rstrip()

        if links:
            new_links_block = f"\n\n{links_header}\n"
            for link in links:
                target = link["target"]
                raw_relation = link.get("relation_type")

                translated_rel = translate_relation(raw_relation, lang)

                new_links_block += f"- {translated_rel}:: [[{target}]]\n"

            new_content = main_text + new_links_block
        else:
            new_content = main_text + "\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        applied_count += 1

    logger.info(f"Classification applied successfully to {applied_count} files.")
