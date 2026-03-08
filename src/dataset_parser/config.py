from dataclasses import dataclass

@dataclass
class ExtractConfig:
    language: str = "ru"
    links_header: str = "## Related Connections"
    keep_prepositions: bool = False
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_threshold: float = 0.5
