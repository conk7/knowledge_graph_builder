import heapq
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set, Tuple

import wikipediaapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Crawler:
    def __init__(
        self,
        seed_topic: str,
        vault_path: str,
        lang: str = "ru",
        links_header: str = "## Related Connections",
        max_pages: int = 100,
        max_depth: int = 3,
        min_score: float = 0.1,
        candidates_limit: int = 100,
        sleep_sec: int = 10,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.seed_topic = seed_topic
        self.vault_path = Path(vault_path)
        self.lang = lang
        self.links_header = links_header
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.min_score = min_score
        self.candidates_limit = candidates_limit
        self.sleep_sec = sleep_sec

        self.visited: Set[str] = set()
        self.saved_titles: Set[str] = set()
        self.raw_outlinks: Dict[str, List[str]] = {}
        self.frontier = []

        logger.info(f"Initializing Cross-Encoder: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, max_length=1024)

        self.wiki_obj = wikipediaapi.Wikipedia(
            language=self.lang, user_agent="WikiGraphBuilder/2.0 (production-ready)"
        )

        self.vault_path.mkdir(parents=True, exist_ok=True)

    def normalize_filename(self, title: str) -> str:
        return title.strip().replace(" ", "_").replace("/", "-").replace('"', "")

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def fetch_summaries_concurrently(self, links: List[str]) -> Dict[str, str]:
        results = {}

        def get_one(title):
            try:
                p = self.wiki_obj.page(title)
                if p.exists():
                    return title, p.summary
            except Exception:
                return None
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = executor.map(get_one, links)

            for res in futures:
                if res:
                    results[res[0]] = res[1]

        return results

    def save_page(
        self, title: str, summary: str, content: str, depth: int, score: float
    ):
        summary = self.clean_text(summary)
        content = self.clean_text(content)

        filename = f"{self.normalize_filename(title)}.md"
        path = self.vault_path / filename

        md_content = f"""{summary}

{content}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.saved_titles.add(title)
        logger.info(
            f"Page saved: '{title}' (filename={filename}) (Score: {score:.4f}, Depth: {depth})"
        )

    def finalize_connections(self):
        logger.info("Starting post-processing of connections...")

        for title in self.saved_titles:
            outlinks = self.raw_outlinks.get(title, [])

            valid_links = [link for link in outlinks if link in self.saved_titles]

            if not valid_links:
                continue
            links_block = f"\n\n{self.links_header}\n"
            links_block += "\n".join(
                [f"- [[{self.normalize_filename(link)}]]" for link in valid_links]
            )

            filename = f"{self.normalize_filename(title)}.md"
            path = self.vault_path / filename

            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(links_block)
            except IOError as e:
                logger.error(f"Failed to append links to {filename}: {e}")

        logger.info(f"Connections updated for {len(self.saved_titles)} files.")

    def rank_candidates(self, candidates: Dict[str, str]) -> List[Tuple[str, float]]:
        if not candidates:
            return []

        pairs = [
            [self.seed_topic, f"{title}: {summary[:500]}"]
            for title, summary in candidates.items()
        ]
        titles = list(candidates.keys())

        try:
            scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return []

        scored_candidates = []
        for i, score in enumerate(scores):
            if score > self.min_score:
                scored_candidates.append((titles[i], score))

        return scored_candidates

    def crawl(self):
        heapq.heappush(self.frontier, (-10.0, 0, self.seed_topic))
        self.visited.add(self.seed_topic)

        while self.frontier and len(self.saved_titles) < self.max_pages:
            neg_score, depth, current_title = heapq.heappop(self.frontier)
            score = -neg_score

            logger.info(f"Processing candidate: '{current_title}' (Score: {score:.2f})")

            try:
                page = self.wiki_obj.page(current_title)
                if not page.exists():
                    logger.warning(f"Page does not exist: {current_title}")
                    continue
            except Exception as e:
                logger.error(f"API Error fetching page {current_title}: {e}")
                time.sleep(self.sleep_sec)
                heapq.heappush(self.frontier, (neg_score, depth, current_title))
                continue

            self.save_page(page.title, page.summary, page.text, depth, score)

            self.raw_outlinks[page.title] = list(page.links.keys())

            if depth >= self.max_depth:
                continue

            raw_links_to_check = [
                t for t in page.links.keys() if ":" not in t and t not in self.visited
            ][: self.candidates_limit]

            if not raw_links_to_check:
                continue

            logger.info(
                f"Fetching summaries for {len(raw_links_to_check)} outgoing links..."
            )

            candidates_with_context = self.fetch_summaries_concurrently(
                raw_links_to_check
            )

            scored_links = self.rank_candidates(candidates_with_context)

            new_added = 0
            for link_title, link_score in scored_links:
                if link_title not in self.visited:
                    self.visited.add(link_title)
                    heapq.heappush(self.frontier, (-link_score, depth + 1, link_title))
                    new_added += 1

            logger.info(f"Added {new_added} new candidates to frontier.")

        self.finalize_connections()
        logger.info(f"Crawling completed. Total saved pages: {len(self.saved_titles)}")


def run_parser(
    seed_topic: str,
    vault_path: str,
    max_pages: int = 100,
    max_depth: int = 3,
    lang: str = "ru",
    links_header: str = "## Related Connections",
):
    crawler = Crawler(
        seed_topic=seed_topic,
        vault_path=vault_path,
        lang=lang,
        links_header=links_header,
        max_pages=max_pages,
        max_depth=max_depth,
    )
    crawler.crawl()
