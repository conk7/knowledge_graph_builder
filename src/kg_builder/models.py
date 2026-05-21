from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class DocumentEntity(BaseModel):
    rel_path: str
    title: str
    aliases: List[str] = Field(default_factory=list)

    @property
    def all_names(self) -> List[str]:
        return [self.title] + self.aliases


class SearchResult(BaseModel):
    text: str
    file_path: str
    distance: float = 0.0


class RerankResult(BaseModel):
    text: str
    score: float


class CandidatePair(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    source_path: Path
    source_content: str
    target_path: Path
    target_content: str
    vector_distance: float = 0.0
    reranker_score: float = 0.0

    @property
    def pair_key(self) -> tuple[Path, Path]:
        return (self.source_path, self.target_path)


class ContextSnippet(BaseModel):
    source_content: str
    target_content: str
    reranker_score: float = 0.0


class GroupedCandidatePair(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    source_path: Path
    target_path: Path
    contexts: List[ContextSnippet]


class NewlyAddedChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    content: str
    file_path: str
