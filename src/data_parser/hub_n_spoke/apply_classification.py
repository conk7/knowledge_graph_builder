import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
                relation_type = link.get("relation_type")

                new_links_block += f"- {relation_type}:: [[{target}]]\n"

            new_content = main_text + new_links_block
        else:
            new_content = main_text + "\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        applied_count += 1

    logger.info(f"Classification applied successfully to {applied_count} files.")
