from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

import src  # noqa: F401 – applies shared logging configuration
from scripts.resolve_neo4j_entities import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    resolve_vault,
)
from src.kg_builder.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_embed_lock = threading.Lock()

IGNORED_DIR_NAMES = {".obsidian", ".kg_builder", "__pycache__"}
DEFAULT_LINK_HEADER_LINE = "## Related Connections"
DEFAULT_BATCH_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", flags=re.DOTALL)
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]")
_DATAVIEW_LINK_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?[^\n:]{1,120}::\s*\[\[[^\]]+\]\]\s*$",
    flags=re.MULTILINE,
)

_TRANSIENT_MARKERS = ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503")


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _parse_retry_delay(exc: Exception, default: float = 30.0) -> float:
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) + 1.0 if m else default


def _to_camel_case_label(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-\s]+", name) if part)


def _normalize_node_id(entity_id: str) -> str:
    return " ".join(entity_id.replace("_", " ").split()).title()


def _prefix_graph_documents(
    graph_documents: list[Any],
    vault: str,
    embed_model: Any = None,
) -> None:
    nodes_to_embed: list[Any] = []
    texts_to_embed: list[str] = []

    for gd in graph_documents:
        if hasattr(gd, "source") and gd.source is not None:
            src_id = getattr(gd.source, "id", None) or ""
            if not str(src_id).startswith(f"{vault}:"):
                gd.source.id = f"{vault}:{src_id}" if src_id else f"{vault}:unknown"
            if not isinstance(getattr(gd.source, "metadata", None), dict):
                gd.source.metadata = {}
            gd.source.metadata.setdefault("vault", vault)

        for node in gd.nodes:
            if not isinstance(node.properties, dict):
                node.properties = {}

            is_document = getattr(node, "type", "").lower() == "document"

            node.properties.setdefault("display_name", node.id)
            node.properties["vault"] = vault

            if not is_document:
                node.id = _normalize_node_id(node.id)
                node.properties["display_name"] = _normalize_node_id(
                    node.properties["display_name"]
                )

            node.id = f"{vault}:{node.id}"

            if embed_model is not None:
                text = (
                    node.properties.get("text")
                    if is_document
                    else node.properties.get("display_name")
                )
                if text:
                    texts_to_embed.append(text)
                    nodes_to_embed.append(node)

        for rel in gd.relationships:
            if not isinstance(rel.properties, dict):
                rel.properties = {}
            rel.properties["vault"] = vault

            src_is_doc = getattr(rel.source, "type", "").lower() == "document"
            tgt_is_doc = getattr(rel.target, "type", "").lower() == "document"

            if not src_is_doc:
                rel.source.id = _normalize_node_id(rel.source.id)
            if not tgt_is_doc:
                rel.target.id = _normalize_node_id(rel.target.id)

            rel.source.id = f"{vault}:{rel.source.id}"
            rel.target.id = f"{vault}:{rel.target.id}"

    if embed_model is not None and texts_to_embed:
        with _embed_lock:
            vectors = embed_model.encode(
                texts_to_embed,
                batch_size=128,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        for node, vec in zip(nodes_to_embed, vectors):
            node.properties["embedding"] = vec.tolist()


def normalize_entity_name(value: str) -> str:
    return " ".join(value.replace("_", " ").split()).casefold()


def to_entity_label(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


def clean_markdown_for_llm(text: str, link_header_line: str | None = None) -> str:
    cleaned = text

    cleaned = _FRONTMATTER_RE.sub("", cleaned)

    header_line = (link_header_line or DEFAULT_LINK_HEADER_LINE).strip()
    if header_line:
        header_re = re.compile(
            rf"^\s*{re.escape(header_line)}\s*$.*\Z",
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        cleaned = header_re.sub("", cleaned)

    cleaned = _DATAVIEW_LINK_RE.sub("", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def configure_logging(level: str) -> None:
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def iter_markdown_files(vault_path: Path) -> Iterable[Path]:
    for path in sorted(vault_path.rglob("*.md")):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        yield path


def read_sample_config(vault_path: Path) -> dict[str, Any]:
    config_path = vault_path / ".kg_builder" / "config.json"
    if not config_path.is_file():
        logger.warning(f"No sample config found at {config_path}")
        return {}

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to parse sample config {config_path}: {exc}")
        return {}


def read_sample_allowed_relationships(config_obj: dict[str, Any]) -> list[str] | None:
    rels = config_obj.get("llm_link_types")
    if not isinstance(rels, list):
        return None

    clean_rels = [str(item).strip() for item in rels if str(item).strip()]
    return clean_rels or None


def read_sample_link_header_line(config_obj: dict[str, Any]) -> str:
    header = config_obj.get("link_header")
    if not isinstance(header, str) or not header.strip():
        return DEFAULT_LINK_HEADER_LINE
    for line in header.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return DEFAULT_LINK_HEADER_LINE


def resolve_vault_paths(
    dataset_root: Path, vault_names: Sequence[str] | None
) -> list[Path]:
    if vault_names:
        paths = [dataset_root / name for name in vault_names]
        missing = [p for p in paths if not p.is_dir()]
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(f"Vault directories not found: {missing_str}")
        return paths

    md_files_at_root = list(iter_markdown_files(dataset_root))
    if md_files_at_root:
        return [dataset_root]

    child_vaults = [
        p
        for p in sorted(dataset_root.iterdir())
        if p.is_dir() and not p.name.startswith(".") and list(iter_markdown_files(p))
    ]
    if not child_vaults:
        raise FileNotFoundError(
            f"No markdown files found in dataset root: {dataset_root}"
        )
    return child_vaults


def load_documents(
    vault_paths: Sequence[Path],
    limit_docs: int | None,
    link_header_line: str | None,
) -> list[Document]:
    documents: list[Document] = []

    for vault_path in vault_paths:
        for md_file in iter_markdown_files(vault_path):
            text = md_file.read_text(encoding="utf-8", errors="replace").strip()
            text = clean_markdown_for_llm(text, link_header_line=link_header_line)
            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(md_file),
                    "source_rel": str(md_file.relative_to(vault_path)),
                    "vault": vault_path.name,
                },
            )
            documents.append(doc)

            if limit_docs and len(documents) >= limit_docs:
                logger.info(f"Reached --limit-docs={limit_docs}")
                return documents

    return documents


def load_single_vault_documents(
    vault_path: Path,
    limit_docs: int | None,
    link_header_line: str | None,
) -> list[Document]:
    return load_documents([vault_path], limit_docs, link_header_line=link_header_line)


def collect_sample_entity_names(vault_path: Path) -> set[str]:
    names: set[str] = set()
    for md_file in iter_markdown_files(vault_path):
        names.add(to_entity_label(md_file.stem))
    return names


def sample_checkpoint_path(state_dir: Path, sample_name: str) -> Path:
    return state_dir / f"{sample_name}.json"


def load_sample_checkpoint(state_dir: Path, sample_name: str) -> set[int]:
    path = sample_checkpoint_path(state_dir, sample_name)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to read checkpoint {path}: {exc}")
        return set()
    done = payload.get("done_batches", []) if isinstance(payload, dict) else []
    return {int(x) for x in done if isinstance(x, int) or str(x).isdigit()}


def save_sample_checkpoint(
    state_dir: Path,
    sample_name: str,
    done_batches: set[int],
    total_batches: int,
) -> None:
    path = sample_checkpoint_path(state_dir, sample_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample": sample_name,
        "done_batches": sorted(done_batches),
        "total_batches": total_batches,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


def batched(seq: Sequence[Document], size: int) -> Iterable[list[Document]]:
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


async def convert_batch_with_retries(
    llm_transformer: LLMGraphTransformer,
    batch: list[Document],
    sample_name: str,
    batch_idx: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[Any]:
    real_failures = 0
    while True:
        try:
            if hasattr(llm_transformer, "aconvert_to_graph_documents"):
                return await llm_transformer.aconvert_to_graph_documents(batch)
            return await asyncio.to_thread(
                llm_transformer.convert_to_graph_documents,
                batch,
            )
        except Exception as exc:
            if _is_transient_error(exc):
                delay = _parse_retry_delay(exc, default=retry_backoff_seconds)
                logger.info(
                    f"[{sample_name}] Batch {batch_idx} rate-limited, retrying in {delay:.1f}s"
                )
                logger.debug(
                    f"[{sample_name}] Batch {batch_idx} rate-limit error: {exc}"
                )
                await asyncio.sleep(delay)
            else:
                real_failures += 1
                logger.warning(
                    f"[{sample_name}] LLM batch {batch_idx} failed attempt {real_failures}/{max_retries}"
                )
                logger.debug(f"[{sample_name}] Batch {batch_idx} error: {exc}")
                if real_failures >= max_retries:
                    raise RuntimeError(
                        f"[{sample_name}] LLM batch {batch_idx} failed after {max_retries} attempts"
                    ) from exc
                await asyncio.sleep(retry_backoff_seconds**real_failures)


async def insert_batch_with_retries(
    graph: Neo4jGraph,
    graph_documents: list[Any],
    sample_name: str,
    batch_idx: int,
    max_retries: int,
    retry_backoff_seconds: float,
    insert_lock: asyncio.Lock,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with insert_lock:
                await asyncio.to_thread(
                    graph.add_graph_documents,
                    graph_documents,
                    baseEntityLabel=True,
                    include_source=True,
                )
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"[{sample_name}] Neo4j insert for batch {batch_idx} failed attempt {attempt}/{max_retries}"
            )
            logger.debug(f"[{sample_name}] Neo4j insert batch {batch_idx} error: {exc}")
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff_seconds**attempt)

    raise RuntimeError(
        f"[{sample_name}] Neo4j insert for batch {batch_idx} failed after {max_retries} attempts"
    ) from last_error


async def process_sample_async(
    sample_name: str,
    graph: Neo4jGraph,
    llm_transformer: LLMGraphTransformer,
    documents: list[Document],
    batch_size: int,
    batch_concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
    state_dir: Path,
    embed_model: Any = None,
) -> int:
    batches = list(batched(documents, batch_size))
    total_batches = len(batches)
    done_batches = load_sample_checkpoint(state_dir, sample_name)
    insert_lock = asyncio.Lock()

    logger.info(
        f"[{sample_name}] Resume checkpoint: {len(done_batches)}/{total_batches} batches already processed"
    )

    pending: list[tuple[int, list[Document]]] = [
        (i, batch_docs)
        for i, batch_docs in enumerate(batches, start=1)
        if i not in done_batches
    ]

    queue: asyncio.Queue[tuple[int, list[Document]]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)

    inserted_docs = 0

    progress = tqdm(
        total=total_batches,
        desc=f"[{sample_name}]",
        unit="batch",
        initial=len(done_batches),
    )

    async def worker() -> None:
        nonlocal inserted_docs, done_batches
        while not queue.empty():
            try:
                batch_idx, batch_docs = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            graph_documents = await convert_batch_with_retries(
                llm_transformer=llm_transformer,
                batch=batch_docs,
                sample_name=sample_name,
                batch_idx=batch_idx,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )

            if not graph_documents:
                logger.warning(
                    f"[{sample_name}] Batch {batch_idx} produced no graph documents"
                )
            else:
                await asyncio.to_thread(
                    _prefix_graph_documents, graph_documents, sample_name, embed_model
                )
                await insert_batch_with_retries(
                    graph=graph,
                    graph_documents=graph_documents,
                    sample_name=sample_name,
                    batch_idx=batch_idx,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    insert_lock=insert_lock,
                )
                inserted_docs += len(graph_documents)
                logger.debug(
                    f"[{sample_name}] Batch {batch_idx} inserted ({len(graph_documents)} graph docs)"
                )

            done_batches.add(batch_idx)
            save_sample_checkpoint(state_dir, sample_name, done_batches, total_batches)
            progress.update(1)

    n_workers = min(max(1, batch_concurrency), len(pending)) if pending else 0
    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(n_workers)])

    progress.close()
    return inserted_docs


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract graph structure from markdown and write to Neo4j"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/test_vaults/gold"),
        help="Dataset root. Can be a single vault directory or a parent folder.",
    )
    parser.add_argument(
        "--vaults",
        nargs="+",
        help="Optional list of vault names under --dataset-root",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for markdown text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for markdown text splitting",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of chunks processed per LLM call batch",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        help="Limit number of source markdown files (useful for smoke tests)",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete all existing nodes/edges in Neo4j before import",
    )
    parser.add_argument(
        "--allowed-nodes",
        nargs="+",
        help="Optional node labels to constrain extraction",
    )
    parser.add_argument(
        "--allowed-relationships",
        nargs="+",
        help="Optional relationship types to constrain extraction. If omitted, uses each sample's .kg_builder/config.json llm_link_types",
    )
    parser.add_argument(
        "--strict-entities",
        action="store_true",
        help="When enabled, constrain extraction to node names derived from markdown file names in each sample",
    )
    parser.add_argument(
        "--batch-concurrency",
        type=int,
        default=DEFAULT_BATCH_CONCURRENCY,
        help="Number of LLM batch conversions running concurrently",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retry attempts for LLM conversion and Neo4j insertion",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Exponential backoff base for retries",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("results/neo4j_ingest_state"),
        help="Directory for per-sample batch checkpoints",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL"),
        help="Google model name for graph extraction",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip computing node embeddings (vector index will not be populated)",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBEDDING_MODEL_NAME,
        help=f"SentenceTransformers model for node embeddings (default: {EMBEDDING_MODEL_NAME})",
    )
    parser.add_argument(
        "--skip-entity-resolution",
        action="store_true",
        help="Skip the post-import semantic entity resolution step",
    )
    parser.add_argument(
        "--er-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for entity resolution (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--er-embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Sentence-Transformers model for entity resolution (default: {DEFAULT_EMBED_MODEL})",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY is required")

    neo4j_url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_password:
        raise EnvironmentError("NEO4J_PASSWORD is required")

    vault_paths = resolve_vault_paths(dataset_root, args.vaults)
    logger.info(f"Resolved vaults: {[str(p) for p in vault_paths]}")

    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)

    if args.clear_existing:
        logger.warning("Clearing existing Neo4j graph data")
        graph.query("MATCH (n) DETACH DELETE n")

    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model=args.model,
        google_api_key=google_api_key,
        convert_system_message_to_human=True,
    )

    embed_model = None
    if not args.skip_embeddings:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading embedding model for vector indexing: {args.embed_model} (device={device})"
        )
        embed_model = SentenceTransformer(args.embed_model, device=device)
        embed_dims = embed_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready — {embed_dims}d vectors")

    total_graph_docs = 0

    for vault_path in vault_paths:
        sample_cfg = read_sample_config(vault_path)
        sample_link_header = read_sample_link_header_line(sample_cfg)

        raw_documents = load_single_vault_documents(
            vault_path,
            args.limit_docs,
            link_header_line=sample_link_header,
        )
        if not raw_documents:
            logger.warning(f"No markdown documents loaded for sample {vault_path}")
            continue

        logger.info(f"[{vault_path.name}] Loaded {len(raw_documents)} markdown files")

        sample_allowed_relationships = args.allowed_relationships
        if not sample_allowed_relationships:
            sample_allowed_relationships = read_sample_allowed_relationships(sample_cfg)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        documents = text_splitter.split_documents(raw_documents)

        if not documents:
            logger.warning(
                f"[{vault_path.name}] No chunks generated after text splitting"
            )
            continue

        _chunk_counter: dict[str, int] = {}
        for doc in documents:
            doc_src = doc.metadata["source"]
            idx = _chunk_counter.get(doc_src, 0)
            _chunk_counter[doc_src] = idx + 1
            doc.metadata["chunk_index"] = idx
            doc.metadata["chunk_id"] = f"{doc_src}::{idx}"
            doc.id = f"{vault_path.name}:{doc.metadata['chunk_id']}"
            if idx > 0:
                doc.metadata["previous_chunk_id"] = f"{doc_src}::{idx - 1}"

        logger.info(
            f"[{vault_path.name}] Prepared {len(documents)} chunks for LLM graph extraction"
        )

        strict_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Knowledge Graph architect parsing personal Obsidian notes "
                    "(mix of English and Russian).\n"
                    "Extract a structured knowledge graph representing the semantic relationships in the text.\n\n"
                    "CRITICAL RULES:\n"
                    "1. EXTRACT TRIPLETS, NOT ISOLATED NODES: Your primary goal is to find relationships BETWEEN entities. "
                    "Do not just list concepts. Every entity MUST be connected to at least one other entity via a relation.\n"
                    "2. INTEGRATE WIKI-LINKS: The text often contains a '## Related Connections' section with links like [[Entity_Name]]. "
                    "You MUST use these as core anchor nodes and connect the tasks/events from the main text to them. "
                    "Example: If text says 'синкануться с науч руком' and has '[[Meeting_Supervisor]]', create a relation: (Научный руководитель) -[PARTICIPATES_IN]-> (Meeting_Supervisor).\n"
                    "3. ENTITY FORMAT: Keep entity names concise (1-4 words). Preserve the original language (RU/EN). "
                    "NEVER extract first-person pronouns ('I', 'Me', 'Я', 'My', 'Author').\n"
                    "4. RELATION FORMAT: Use descriptive, uppercase English verbs with underscores for relations "
                    "(e.g., REQUIRES, USES_TOOL, RELATES_TO, INCLUDES, DEPENDS_ON, WATCHED).\n"
                    "5. IMPLICIT LOGIC: Infer logical connections. If a list of tasks is under 'Diploma_Project', connect each task to the project with [PART_OF] or [RELATES_TO].\n",
                ),
                ("human", "Text to extract from:\n{input}"),
            ]
        )

        _FALLBACK_ALLOWED_NODES = [
            "Concept",
            "Algorithm",
            "Technology",
            "Task",
            "Person",
            "Organization",
            "Document",
            "Tool",
            "Концепция",
            "Алгоритм",
            "Технология",
            "Задача",
            "Персона",
            "Организация",
            "Документ",
            "Инструмент",
        ]

        transformer_kwargs: dict[str, object] = {"llm": llm, "prompt": strict_prompt}
        if args.strict_entities:
            sample_entities = sorted(collect_sample_entity_names(vault_path))
            transformer_kwargs["allowed_nodes"] = sample_entities
            logger.info(
                f"[{vault_path.name}] Strict entity mode: constrained to {len(sample_entities)} markdown-derived nodes"
            )
        elif args.allowed_nodes:
            transformer_kwargs["allowed_nodes"] = args.allowed_nodes
        else:
            transformer_kwargs["allowed_nodes"] = _FALLBACK_ALLOWED_NODES
            logger.info(
                f"[{vault_path.name}] Using fallback generic ontology ({len(_FALLBACK_ALLOWED_NODES)} node types)"
            )
        if sample_allowed_relationships:
            transformer_kwargs["allowed_relationships"] = sample_allowed_relationships

        llm_transformer = LLMGraphTransformer(**transformer_kwargs)

        inserted_for_sample = asyncio.run(
            process_sample_async(
                sample_name=vault_path.name,
                graph=graph,
                llm_transformer=llm_transformer,
                documents=documents,
                batch_size=args.batch_size,
                batch_concurrency=args.batch_concurrency,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                state_dir=args.state_dir,
                embed_model=embed_model,
            )
        )
        total_graph_docs += inserted_for_sample
        logger.info(
            f"[{vault_path.name}] Sample completed: inserted {inserted_for_sample} graph documents"
        )

        vault_label = _to_camel_case_label(vault_path.name)
        graph.query(
            f"MATCH (n) WHERE n.vault = $vault SET n:{vault_label}",
            params={"vault": vault_path.name},
        )
        logger.info(
            f"[{vault_path.name}] Assigned label :{vault_label} to all vault nodes"
        )
        graph.query(
            f"CREATE INDEX {vault_label}_id IF NOT EXISTS FOR (n:{vault_label}) ON (n.id)"
        )
        graph.query(
            f"CREATE FULLTEXT INDEX {vault_label}_text_index IF NOT EXISTS "
            f"FOR (n:{vault_label}) ON EACH [n.display_name, n.text]"
        )
        graph.query(
            "CREATE FULLTEXT INDEX document_text_index IF NOT EXISTS "
            "FOR (d:Document) ON EACH [d.text]"
        )
        logger.info(
            f"[{vault_path.name}] Created B-tree ID index and fulltext indexes for :{vault_label}"
        )
        if embed_model is not None:
            graph.query(
                f"CREATE VECTOR INDEX {vault_label}_vector_index IF NOT EXISTS "
                f"FOR (n:{vault_label}) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {embed_dims}, "
                f"`vector.similarity_function`: 'cosine'"
                f"}}}}"
            )
            logger.info(
                f"[{vault_path.name}] Created vector index :{vault_label}_vector_index "
                f"({embed_dims}d cosine)"
            )
        graph.query(
            "CREATE INDEX document_chunk_id IF NOT EXISTS FOR (d:Document) ON (d.chunk_id)"
        )
        graph.query(
            "CREATE INDEX document_prev_chunk_id IF NOT EXISTS FOR (d:Document) ON (d.previous_chunk_id)"
        )
        logger.info(f"[{vault_path.name}] Created indexes for Document chunk IDs")
        graph.query(
            """
            MATCH (c1:Document), (c2:Document)
            WHERE c1.vault = $vault AND c2.vault = $vault
              AND c1.chunk_id IS NOT NULL AND c2.previous_chunk_id IS NOT NULL
              AND c1.chunk_id = c2.previous_chunk_id
            MERGE (c1)-[:NEXT_CHUNK]->(c2)
            """,
            params={"vault": vault_path.name},
        )
        logger.info(f"[{vault_path.name}] Linked sequential chunks with [:NEXT_CHUNK]")

    logger.info(
        f"Graph build finished. Inserted {total_graph_docs} graph documents into Neo4j ({neo4j_url})"
    )

    if not args.skip_entity_resolution:
        if embed_model is not None and args.er_embed_model == args.embed_model:
            er_embed_model = embed_model
            logger.info(
                f"Reusing already-loaded embedding model for entity resolution: {args.er_embed_model}"
            )
        else:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"Loading embedding model for entity resolution: {args.er_embed_model} (device={device})"
            )
            er_embed_model = SentenceTransformer(args.er_embed_model, device=device)

        total_merged = 0
        for vault_path in vault_paths:
            total_merged += resolve_vault(
                graph, vault_path.name, er_embed_model, args.er_threshold
            )
        logger.info(f"Entity resolution finished — total nodes merged: {total_merged}")
    else:
        logger.info("Entity resolution skipped (--skip-entity-resolution)")


if __name__ == "__main__":
    main()
