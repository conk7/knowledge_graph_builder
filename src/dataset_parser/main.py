import argparse
import logging
from pathlib import Path

from .apply_classification import apply_classification
from .config import ExtractConfig
from .extract_contexts import extract_contexts
from .process_links import process_links_with_wikidata
from .wiki_parser import run_parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Dataset Parser Pipeline")

    # Wiki Parser Args
    parser.add_argument(
        "--seed",
        type=str,
        default="Машинное обучение",
        help="Seed topic for Wikipedia crawling",
    )
    parser.add_argument(
        "--lang", type=str, default="ru", help="Language code (ru, en, etc.)"
    )
    parser.add_argument(
        "--links-header",
        type=str,
        default="## Related Connections",
        help="Header for links section in markdown files",
    )
    parser.add_argument(
        "--vault", type=str, required=True, help="Path to save/read markdown vault"
    )
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to crawl")
    parser.add_argument("--max-depth", type=int, default=3, help="Max crawl depth")

    # Context Extraction Args
    parser.add_argument(
        "--contexts-output",
        type=str,
        default="link_contexts.json",
        help="Output JSON for contexts",
    )
    parser.add_argument(
        "--keep-prepositions",
        action="store_true",
        help="Keep prepositions in lemma matching",
    )
    parser.add_argument(
        "--use-cross-encoder",
        action="store_true",
        help="Use cross-encoder for matching",
    )
    parser.add_argument(
        "--ce-model", type=str, default=None, help="Cross-encoder model name"
    )
    parser.add_argument(
        "--ce-threshold", type=float, default=None, help="Cross-encoder threshold"
    )

    # Link Filter Args
    parser.add_argument(
        "--final-output",
        type=str,
        default="classified_links.json",
        help="Final output JSON with Wikidata relations",
    )
    parser.add_argument(
        "--use-llm-fallback",
        type=bool,
        default=False,
        help="Use LLM for link classification if they are not classified in wiki",
    )

    # Pipeline Control
    parser.add_argument(
        "--skip-crawl", action="store_true", help="Skip Wikipedia crawling step"
    )
    parser.add_argument(
        "--skip-extract", action="store_true", help="Skip context extraction step"
    )
    parser.add_argument(
        "--skip-filter", action="store_true", help="Skip Wikidata filtering step"
    )
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="Skip applying classifications back to markdown",
    )

    args = parser.parse_args()

    vault_path = Path(args.vault)
    contexts_json = Path(args.contexts_output)
    final_json = Path(args.final_output)

    # 1. Wikipedia Crawling
    if not args.skip_crawl:
        logger.info("Starting Step 1: Wikipedia Crawling...")
        run_parser(
            seed_topic=args.seed,
            vault_path=vault_path,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            lang=args.lang,
            links_header=args.links_header,
        )
    else:
        logger.info("Skipping Step 1: Wikipedia Crawling")

    # 2. Context Extraction
    if not args.skip_extract:
        logger.info("Starting Step 2: Context Extraction...")

        # Build config
        config = ExtractConfig(
            language=args.lang,
            links_header=args.links_header,
            keep_prepositions=args.keep_prepositions,
            use_cross_encoder=args.use_cross_encoder,
        )
        if args.ce_model:
            config.cross_encoder_model = args.ce_model
        if args.ce_threshold is not None:
            config.cross_encoder_threshold = args.ce_threshold

        extract_contexts(
            vault_path=vault_path, output_json=contexts_json, config=config
        )
    else:
        logger.info("Skipping Step 2: Context Extraction")

    # 3. Wikidata Filtering/Classification
    if not args.skip_filter:
        logger.info("Starting Step 3: Wikidata Filtering...")
        if not contexts_json.exists():
            logger.error(f"Contexts file {contexts_json} not found. Cannot run filter.")
        else:
            process_links_with_wikidata(
                input_file=str(contexts_json),
                output_file=str(final_json),
                lang=args.lang,
                use_llm_fallback=args.use_llm_fallback,
            )
    else:
        logger.info("Skipping Step 3: Wikidata Filtering")

    # 4. Apply Classifications
    if not args.skip_apply:
        logger.info("Starting Step 4: Applying Classifications to Markdown...")
        if not final_json.exists():
            logger.error(f"Classified JSON file {final_json} not found. Cannot apply.")
        else:
            apply_classification(
                vault_path=vault_path,
                classified_json=final_json,
                links_header=args.links_header,
                lang=args.lang,
            )
    else:
        logger.info("Skipping Step 4: Applying Classifications")

    logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    main()
