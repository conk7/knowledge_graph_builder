from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import wikipediaapi
import os
import re
import logging
import heapq
from typing import List, Dict, Tuple, Set
from sentence_transformers import CrossEncoder

LANG = "ru"
VAULT_PATH = "./Test_Vault"
SEED_TOPIC = "Искусственный интеллект"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_PAGES = 30
MAX_DEPTH = 3
MIN_SCORE = 0.1
CANDIDATES_LIMIT = 20

MMR_DIVERSITY = 0.4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

os.makedirs(VAULT_PATH, exist_ok=True)


class Crawler:
    def __init__(self, seed_topic: str):
        self.seed_topic = seed_topic

        self.visited: Set[str] = set()

        self.saved_titles: Set[str] = set()

        self.raw_outlinks: Dict[str, List[str]] = {}

        self.frontier = []

        logger.info(f"Initializing Cross-Encoder: {RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=1024)

        logger.info(f"Loading Bi-Encoder: {EMBED_MODEL_NAME}")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.wiki_obj = wikipediaapi.Wikipedia(
            language=LANG, user_agent="WikiGraphBuilder/2.0 (production-ready)"
        )

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
        path = os.path.join(VAULT_PATH, filename)

        md_content = f"""# {title}

{summary}

{content}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.saved_titles.add(title)
        logger.info(f"Page saved: '{title}' (Score: {score:.4f}, Depth: {depth})")

    def finalize_connections(self):
        logger.info("Starting post-processing of connections...")

        for title in self.saved_titles:
            outlinks = self.raw_outlinks.get(title, [])

            valid_links = [link for link in outlinks if link in self.saved_titles]

            if not valid_links:
                continue
            links_block = "\n\n## Related Connections\n"
            links_block += "\n".join(
                [f"- [[{self.normalize_filename(link)}]]" for link in valid_links]
            )

            filename = f"{self.normalize_filename(title)}.md"
            path = os.path.join(VAULT_PATH, filename)

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
            if score > MIN_SCORE:
                scored_candidates.append((titles[i], score))

        return scored_candidates

    def crawl(self):
        heapq.heappush(self.frontier, (-10.0, 0, self.seed_topic))
        self.visited.add(self.seed_topic)

        while self.frontier and len(self.saved_titles) < MAX_PAGES:
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
                continue

            self.save_page(page.title, page.summary, page.text, depth, score)

            self.raw_outlinks[page.title] = list(page.links.keys())

            if depth >= MAX_DEPTH:
                continue

            raw_links_to_check = [
                t for t in page.links.keys() if ":" not in t and t not in self.visited
            ][:CANDIDATES_LIMIT]

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


if __name__ == "__main__":
    crawler = Crawler(SEED_TOPIC)
    crawler.crawl()
