import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import frontmatter

from .config import LINK_HEADER
from .models import DocumentEntity

logger = logging.getLogger(__name__)


class VaultManager:
    def __init__(
        self, vault_path: Path, ignored_dirs: List[Path], link_header: str = LINK_HEADER
    ):
        if not vault_path.is_dir():
            raise FileNotFoundError(f"Invalid vault path: {vault_path}")
        self.vault_path = vault_path
        self.ignored_dirs = [p.resolve() for p in ignored_dirs]
        self.link_header = link_header
        logger.info(f"VaultManager initiated for : {self.vault_path}")

    def _is_path_ignored(self, path: Path) -> bool:
        resolved_path = path.resolve()
        for ignored in self.ignored_dirs:
            if ignored in resolved_path.parents or resolved_path == ignored:
                return True
        return False

    def _split_content_and_links(self, content: str) -> Tuple[str, str]:
        if self.link_header in content:
            parts = content.split(self.link_header)
            return parts[0].rstrip(), parts[-1].strip()
        return content, ""

    def scan_markdown_files(self) -> List[Path]:
        logger.info("Scanning vault...")
        all_md_files = list(self.vault_path.rglob("*.md"))

        valid_files = []
        for file_path in all_md_files:
            if not self._is_path_ignored(file_path):
                valid_files.append(file_path)

        logger.info(f"Found {len(valid_files)} files")
        return valid_files

    def build_global_entity_dict(
        self, files: Optional[List[Path]] = None
    ) -> Dict[str, DocumentEntity]:
        entity_dict = {}
        for file_path in (files if files is not None else self.scan_markdown_files()):
            rel_path = str(file_path.relative_to(self.vault_path))
            title = file_path.stem

            try:
                post = frontmatter.load(file_path)
                aliases = post.get("aliases", [])

                if isinstance(aliases, str):
                    aliases = [a.strip() for a in aliases.split(",") if a.strip()]
                elif isinstance(aliases, list):
                    aliases = [str(a).strip() for a in aliases if str(a).strip()]
                else:
                    aliases = []
            except Exception as e:
                logger.error(f"Failed to parse frontmatter from {file_path}: {e}")
                aliases = []

            entity_dict[rel_path] = DocumentEntity(
                rel_path=rel_path, title=title, aliases=aliases
            )

        return entity_dict

    def get_file_content(self, file_path: Path) -> str:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return ""

    @staticmethod
    def calculate_hash_from_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def calculate_file_hash(self, file_path: Path) -> str:
        content = self.get_file_content(file_path)
        return self.calculate_hash_from_content(content)

    def append_links_to_file(self, file_path: Path, new_links: Set[str]):
        if not new_links:
            return

        content = self.get_file_content(file_path)
        main_content, links_section = self._split_content_and_links(content)

        existing_links = set(
            line.strip() for line in links_section.split("\n") if line.strip()
        )

        links_to_add = new_links - existing_links

        if not links_to_add:
            logger.info(f"File already contains all the links {file_path.name}")
            return

        all_links = sorted(existing_links | links_to_add)
        links_text = "\n".join(all_links) + "\n"

        try:
            with file_path.open("w", encoding="utf-8") as f:
                f.write(main_content + self.link_header + links_text)
            logger.info(f"Added {len(links_to_add)} new links to {file_path.name}")
        except Exception as e:
            logger.error(f"Could not add new links to {file_path}: {e}")

    def move_file(self, source_path: Path, destination_dir: Path):
        if not source_path.exists():
            logger.warning(f"Could not find file: {source_path}")
            return

        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / source_path.name
        source_path.rename(destination_path)
        logger.info(f"Moved {source_path.name} to {destination_dir}")

    def ensure_dir_exists(self, dir_path: Path):
        dir_path.mkdir(parents=True, exist_ok=True)

    def clear_all_ai_links(self, files_to_clear: List[Path]):
        logger.info(f"Clearing AI-generated links from {len(files_to_clear)} files...")
        cleared_count = 0
        for file_path in files_to_clear:
            try:
                content = self.get_file_content(file_path)
                main_content, links_section = self._split_content_and_links(content)
                if links_section:
                    with file_path.open("w", encoding="utf-8") as f:
                        f.write(main_content + "\n")
                    cleared_count += 1
            except Exception as e:
                logger.error(f"Failed to clear links from {file_path}: {e}")
        logger.info(f"Links cleared from {cleared_count} files.")

    def determine_changes(
        self, metadata_manager
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        logger.info("Looking for changes in vault...")
        files_paths = self.scan_markdown_files()
        files_rel_paths = {p.relative_to(self.vault_path) for p in files_paths}

        tracked_files_rel = set(map(Path, metadata_manager.vault.files.keys()))

        added_files = [
            self.vault_path / p for p in (files_rel_paths - tracked_files_rel)
        ]
        removed_files = [
            self.vault_path / p for p in (tracked_files_rel - files_rel_paths)
        ]

        potential_updates = files_rel_paths.intersection(tracked_files_rel)
        updated_files = []
        for file_rel in potential_updates:
            file_abs = self.vault_path / file_rel
            if not file_abs.exists():
                continue
            record = metadata_manager.get_file_record(str(file_rel))
            if record is None:
                updated_files.append(file_abs)
                continue
            new_hash = self.calculate_file_hash(file_abs)
            if record.hash != new_hash:
                updated_files.append(file_abs)

        logger.info(
            f"Found: {len(added_files)} new, {len(updated_files)} updated, {len(removed_files)} deleted files."
        )
        return added_files, updated_files, removed_files

    def process_removals(
        self, files_to_process: List[Path], metadata_manager, vector_store
    ):
        if not files_to_process:
            return
        logger.info(f"Processing {len(files_to_process)} updated/deleted files...")

        for file_path in files_to_process:
            rel_path_str = str(file_path.relative_to(self.vault_path))
            metadata_manager.remove_file_record(rel_path_str)
            vector_store.remove_document(rel_path_str)

    def process_additions_and_updates(
        self, files_to_process: List[Path], metadata_manager, vector_store
    ):
        if not files_to_process:
            return []
        logger.info(f"Indexing/Retrieving {len(files_to_process)} files...")

        from .models import NewlyAddedChunk

        newly_added_chunks = []
        for file_path in files_to_process:
            rel_path_str = str(file_path.relative_to(self.vault_path))
            content = self.get_file_content(file_path)
            if not content:
                continue

            main_content, _ = self._split_content_and_links(content)
            file_hash = self.calculate_hash_from_content(content)
            chunk_texts = vector_store.get_and_update_file_chunks(
                rel_path_str, main_content, file_hash
            )

            if not chunk_texts:
                continue

            for chunk_text in chunk_texts:
                newly_added_chunks.append(
                    NewlyAddedChunk(content=chunk_text, file_path=rel_path_str)
                )

            from .metadata_manager import FileMetadata

            file_meta = FileMetadata(file_path=rel_path_str, hash=file_hash)
            metadata_manager.add_or_update_file_record(file_meta)
            logger.info(f"Processed '{file_path.name}' ({len(chunk_texts)} chunks).")

        if newly_added_chunks and vector_store.vector_weight < 1.0:
            vector_store.rebuild_fts_index()

        return newly_added_chunks
