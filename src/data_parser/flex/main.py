import argparse
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import wikipediaapi

from src.data_parser.hub_n_spoke.extract_contexts import (
    batch_extract_lemmas_and_heads,
    check_match_with_proximity,
    load_nlp,
    process_text_sentences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def normalize_filename(title: str) -> str:
    clean_title = title.strip().replace(" ", "_").replace("/", "-")
    clean_title = clean_title.replace('"', "").replace("'", "")
    return clean_title


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_stop_sections(text: str) -> str:
    stop_sections = [
        "См. также",
        "Литература",
        "Примечания",
        "Ссылки",
        "See also",
        "References",
        "Notes",
        "External links",
        "Bibliography",
        "Further reading",
    ]

    stop_pattern_raw = (
        r"(?mi)^(?:#{1,6}\s+|={1,6}\s*)?("
        + "|".join(map(re.escape, stop_sections))
        + r")(?:\s*[:\-])?\s*$"
    )
    stop_regex = re.compile(stop_pattern_raw)

    min_idx = len(text)
    for match in stop_regex.finditer(text):
        idx = match.start()
        if idx < min_idx:
            min_idx = idx

    return text[:min_idx].strip()


def filter_links_by_text(titles: List[str], text: str, lang: str) -> List[str]:
    logger.info(f"Filtering {len(titles)} links by main text (language: {lang})...")
    nlp = load_nlp(lang)
    sentences_info = process_text_sentences(text, nlp)

    phrase_results = batch_extract_lemmas_and_heads(titles, nlp)

    valid_titles = []
    for i, title in enumerate(titles):
        lemmas, head = phrase_results[i]
        match_found = False
        for sent_info in sentences_info:
            if check_match_with_proximity(lemmas, head, sent_info["lemmas"]):
                match_found = True
                break
        if match_found:
            valid_titles.append(title)

    logger.info(
        f"Filtering complete. {len(valid_titles)} links remaining out of {len(titles)}."
    )
    return valid_titles


def fetch_and_save_page(
    wiki: wikipediaapi.Wikipedia,
    title: str,
    output_dir: Path,
    max_retries: int = 3,
    filter_links: bool = False,
    lang: str = "ru",
) -> Tuple[str, List[str]]:
    page = None
    attempt = 0

    while attempt < max_retries:
        try:
            logger.info(f"Requesting article '{title}' (attempt {attempt + 1})...")
            page = wiki.page(title)

            if not page.exists():
                logger.error(f"Article '{title}' not found.")
                return None, []
            break
        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                wait_time = attempt * 5
                logger.warning(
                    f"Error requesting '{title}': {e}. Waiting {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to fetch article '{title}' after {max_retries} attempts."
                )
                return None, []

    full_text = strip_stop_sections(clean_text(page.text))

    links = [t for t in page.links.keys() if ":" not in t]
    if filter_links:
        links = filter_links_by_text(links, full_text, lang)

    content_parts = [full_text]
    md_content = "\n\n".join(content_parts) + "\n"

    filename = f"{normalize_filename(page.title)}.md"
    file_path = output_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Article '{page.title}' saved to: {file_path}")
    return page.title, links


def finalize_connections(
    output_dir: Path,
    raw_outlinks: Dict[str, List[str]],
    links_header: str = "## Related Connections",
    remove_missing: bool = True,
) -> None:
    logger.info("Finalizing connection processing...")

    existing_filenames = {f.stem.lower() for f in output_dir.glob("*.md")}

    for title, links in raw_outlinks.items():
        if remove_missing:
            initial_count = len(links)
            links = [
                t for t in links if normalize_filename(t).lower() in existing_filenames
            ]
            if initial_count > len(links):
                logger.debug(
                    f"[{title}] Removed {initial_count - len(links)} missing links."
                )

        if links:
            links_list_str = "\n".join(
                [f"- [[{normalize_filename(link)}]]" for link in sorted(links)]
            )
        else:
            links_list_str = "_No links found._"

        links_block = f"\n{links_header}\n\n{links_list_str}\n"

        filename = f"{normalize_filename(title)}.md"
        file_path = output_dir / filename

        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(links_block)
        except IOError as e:
            logger.error(f"Failed to append links to {filename}: {e}")

    logger.info(f"Connections updated for {len(raw_outlinks)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Downloads Wikipedia articles to Markdown with link filtering."
    )
    parser.add_argument("titles", nargs="+", help="Article titles")
    parser.add_argument("--lang", default="ru", help="Language (ru, en)")
    parser.add_argument("--out", default="vault", help="Output directory")
    parser.add_argument(
        "--header", default="## Related Connections", help="Header for links"
    )
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument(
        "--filter-links",
        action="store_true",
        help="Only keep links mentioned in the article's main text",
    )
    parser.add_argument(
        "--remove-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove links to articles that are not in the folder (and are not being downloaded now)",
    )

    args = parser.parse_args()

    wiki = wikipediaapi.Wikipedia(
        language=args.lang,
        user_agent="WikiPageFetcher/1.0 (Diploma Project; mailto:example@example.com)",
    )

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_outlinks = {}

    for title in args.titles:
        actual_title, links = fetch_and_save_page(
            wiki=wiki,
            title=title,
            output_dir=out_path,
            max_retries=args.retries,
            filter_links=args.filter_links,
            lang=args.lang,
        )
        if actual_title:
            raw_outlinks[actual_title] = links

    finalize_connections(
        output_dir=out_path,
        raw_outlinks=raw_outlinks,
        links_header=args.header,
        remove_missing=args.remove_missing,
    )


if __name__ == "__main__":
    main()
