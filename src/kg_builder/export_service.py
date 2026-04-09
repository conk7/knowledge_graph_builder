import json
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Dict, Set

from .config import META_DIR_NAME
from .vault_manager import VaultManager

logger = logging.getLogger(__name__)


class SaveMode(str, Enum):
    INPLACE = "inplace"
    JSON = "json"
    EXPORT = "export"


class ExportService:
    def __init__(
        self,
        vault_path: Path,
        vault_manager: VaultManager,
        export_path: Path,
        output_json_path: Path,
    ):
        self.vault_path = vault_path
        self.vault_manager = vault_manager
        self.export_path = export_path
        self.output_json_path = output_json_path

    def save_new_links(self, links_to_write: Dict[Path, Set[str]], save_mode: SaveMode):
        logger.info(f"Saving new links (Mode: {save_mode.value})...")

        if save_mode == SaveMode.INPLACE:
            for rel_path_str, links in links_to_write.items():
                file_abs_path = self.vault_path / rel_path_str
                self.vault_manager.append_links_to_file(file_abs_path, links)
        elif save_mode == SaveMode.JSON:
            self._save_links_to_json(links_to_write)
        elif save_mode == SaveMode.EXPORT:
            self._export_enriched_vault(links_to_write)

    def _save_links_to_json(self, links_to_write: Dict[Path, Set[str]]):
        logger.info(f"Saving new links to JSON: {self.output_json_path}")
        serializable_links = {
            str(path): list(links) for path, links in links_to_write.items()
        }
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_links, f, ensure_ascii=False, indent=4)

    def _export_enriched_vault(self, links_to_write: Dict[Path, Set[str]]):
        logger.info(f"Exporting enriched vault to: {self.export_path}")

        if self.export_path.exists():
            logger.info(f"Removing existing export directory: {self.export_path}")
            shutil.rmtree(self.export_path)

        shutil.copytree(
            self.vault_path,
            self.export_path,
            ignore=shutil.ignore_patterns(META_DIR_NAME, ".obsidian", "*.log"),
        )

        exported_md_files = list(self.export_path.rglob("*.md"))
        self.vault_manager.clear_all_ai_links(exported_md_files)

        for rel_path_str, links in links_to_write.items():
            exported_file_path = self.export_path / rel_path_str
            if exported_file_path.exists():
                self.vault_manager.append_links_to_file(exported_file_path, links)
