import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_LOG_LEVEL, setup_logging
from .graph_builder import KnowledgeGraphBuilder, SaveMode

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] not in ["init", "run"] and not sys.argv[1].startswith("-"):
            sys.argv.insert(1, "run")
    elif len(sys.argv) == 1:
        sys.argv.append("run")

    parser = argparse.ArgumentParser(
        description="Knowledge Graph Builder: Semantic Linker for Obsidian Vaults"
    )

    subparsers = parser.add_subparsers(dest="command", help="Mode of operation")

    init_parser = subparsers.add_parser(
        "init", help="Initialize AI metadata and configuration in the vault"
    )
    init_parser.add_argument(
        "vault",
        type=str,
        nargs="?",
        default=".",
        help="Path to the Obsidian vault to initialize (default: current directory)",
    )
    init_parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    run_parser = subparsers.add_parser(
        "run", help="Run the Knowledge Graph update and semantic linking"
    )
    run_parser.add_argument(
        "vault",
        type=str,
        nargs="?",
        default=".",
        help="Path to the Obsidian vault to process (default: current directory)",
    )
    run_parser.add_argument(
        "--save-mode",
        type=str,
        choices=[mode.value for mode in SaveMode],
        default=SaveMode.INPLACE.value,
        help="How to save identified links (inplace, json, export)",
    )
    run_parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Start with a fresh index, clearing previous AI-generated links and metadata",
    )
    run_parser.add_argument(
        "--api",
        action="store_true",
        help="Use an external API for LLM classification instead of a local model",
    )
    run_parser.add_argument(
        "--export-path",
        type=str,
        help="Target path for exporting the enriched vault (only used in 'export' mode)",
    )
    run_parser.add_argument(
        "--output-json",
        type=str,
        help="Path for saving the JSON output of links (only used in 'json' mode)",
    )
    run_parser.add_argument(
        "--ignored-dirs",
        nargs="+",
        help="List of directories within the vault to ignore",
    )
    run_parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    if not args.command:
        args.command = "run"

    vault_path = Path(args.vault).resolve()
    if not vault_path.is_dir():
        print(f"Error: Vault path '{vault_path}' does not exist or is not a directory.")
        return

    setup_logging(vault_path, log_level=args.log_level)

    logger.info(f"Starting Knowledge Graph Builder in '{args.command}' mode...")

    try:
        if args.command == "init":
            builder = KnowledgeGraphBuilder(vault_path=vault_path)
            builder.initialize_vault()
            logger.info("Initialization complete.")
            return

        ignored_paths = []
        if args.ignored_dirs:
            for p in args.ignored_dirs:
                path = Path(p)
                if not path.is_absolute():
                    path = vault_path / path
                ignored_paths.append(path.resolve())

        save_mode = SaveMode(args.save_mode)
        export_path = Path(args.export_path).resolve() if args.export_path else None
        output_json_path = (
            Path(args.output_json).resolve() if args.output_json else None
        )

        builder = KnowledgeGraphBuilder(
            vault_path=vault_path,
            ignored_dirs=ignored_paths,
            fresh_start=args.fresh_start,
            use_api=args.api,
            save_mode=save_mode,
            export_path=export_path,
            output_json_path=output_json_path,
        )

        builder.run_update()
        builder.close()

        logger.info("Processing complete.")

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
