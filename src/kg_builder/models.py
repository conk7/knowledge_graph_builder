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


class LinkPrediction(BaseModel):
    """Represents a single LLM prediction for a relationship between two chunks."""

    relation_type: str
    reasoning: str
    text_a: str
    text_b: str
    reranker_score: float = 0.0
    vector_distance: float = 0.0


class LinkConflict(BaseModel):
    """Represents a conflict between multiple chunk-level link predictions for the same two files."""

    filename_a: str
    filename_b: str
    candidate_counts: dict[str, int]
    evidence: list[LinkPrediction]


class NewlyAddedChunk(BaseModel):
    content: str
    file_path: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
