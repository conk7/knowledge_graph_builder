from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class DocumentEntity(BaseModel):
    """Represents a Markdown document entity with its title and known aliases."""

    rel_path: str
    title: str
    aliases: List[str] = Field(default_factory=list)

    @property
    def all_names(self) -> List[str]:
        return [self.title] + self.aliases


class SearchResult(BaseModel):
    """Represents a text chunk retrieved from the vector store."""

    text: str
    file_path: str
    distance: float = 0.0


class RerankResult(BaseModel):
    """Represents a text chunk reranked with a cross-encoder score."""

    text: str
    score: float


class CandidatePair(BaseModel):
    """Represents a pair of source chunk and a potentially related target chunk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: Path
    source_content: str
    target_path: Path
    target_content: str
    vector_distance: float = 0.0
    reranker_score: float = 0.0

    @property
    def pair_key(self) -> tuple[Path, Path]:
        """A unique tuple key for this specific file connection."""
        return (self.source_path, self.target_path)


class ContextSnippet(BaseModel):
    """A single (source_chunk, target_chunk) evidence pair for a file-to-file link candidate."""

    source_content: str
    target_content: str
    reranker_score: float = 0.0


class GroupedCandidatePair(BaseModel):
    """All context snippets for a unique (source_path, target_path) file pair, ranked by score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: Path
    target_path: Path
    contexts: List[ContextSnippet]


class NewlyAddedChunk(BaseModel):
    content: str
    file_path: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
