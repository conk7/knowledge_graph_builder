import hashlib
import logging
from pathlib import Path
from typing import List, Set

from config import LINK_HEADER


class VaultManager:
    def __init__(self, vault_path: Path, ignored_dirs: List[Path]):
        if not vault_path.is_dir():
            raise FileNotFoundError(f"Invalid vault path: {vault_path}")
        self.vault_path = vault_path
        self.ignored_dirs = [p.resolve() for p in ignored_dirs]
        logging.info(f"VaultManager initiated for : {self.vault_path}")

    def _is_path_ignored(self, path: Path) -> bool:
        resolved_path = path.resolve()
        for ignored in self.ignored_dirs:
            if ignored in resolved_path.parents or resolved_path == ignored:
                return True
        return False

    def scan_markdown_files(self) -> List[Path]:
        logging.info("Scanning vault...")
        all_md_files = list(self.vault_path.rglob("*.md"))

        valid_files = []
        for file_path in all_md_files:
            if not self._is_path_ignored(file_path):
                valid_files.append(file_path)

        logging.info(f"Found {len(valid_files)} files")
        return valid_files

    def get_file_content(self, file_path: Path) -> str:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Could not read {file_path}: {e}")
            return ""

    def calculate_file_hash(self, file_path: Path) -> str:
        content = self.get_file_content(file_path)
        return hashlib.sha256(content.encode()).hexdigest()

    def append_links_to_file(self, file_path: Path, new_links: Set[str]):
        if not new_links:
            return

        content = self.get_file_content(file_path)
        links_header = LINK_HEADER

        existing_links = set()
        if links_header in content:
            links_section = content.split(links_header)[-1]
            existing_links.update(
                line.strip() for line in links_section.split("\n") if line.strip()
            )

        links_to_add = new_links - existing_links

        if not links_to_add:
            logging.info(f"File already contains all the links {file_path.name}")
            return

        append_text = ""
        if links_header not in content:
            append_text += links_header

        append_text += "\n".join(sorted(list(links_to_add))) + "\n"

        try:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(append_text)
            logging.info(f"Added {len(links_to_add)} new links to {file_path.name}")
        except Exception as e:
            logging.error(f"Could not add new links to {file_path}: {e}")

    def move_file(self, source_path: Path, destination_dir: Path):
        if not source_path.exists():
            logging.warning(f"Could not find file: {source_path}")
            return

        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / source_path.name
        source_path.rename(destination_path)
        logging.info(f"Moved {source_path.name} to {destination_dir}")

    def ensure_dir_exists(self, dir_path: Path):
        dir_path.mkdir(parents=True, exist_ok=True)

    def clear_all_ai_links(self, files_to_clear: List[Path]):
        links_header = LINK_HEADER
        logging.info(f"Clearing AI-generated links from {len(files_to_clear)} files...")
        cleared_count = 0
        for file_path in files_to_clear:
            try:
                content = self.get_file_content(file_path)
                if links_header in content:
                    cleaned_content = content.split(links_header)[0].rstrip()
                    with file_path.open("w", encoding="utf-8") as f:
                        f.write(cleaned_content)
                    cleared_count += 1
            except Exception as e:
                logging.error(f"Failed to clear links from {file_path}: {e}")
        logging.info(f"Links cleared from {cleared_count} files.")
