from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-large"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str], prefix: str) -> np.ndarray:
    """E5 wants prefixes: 'query: ' for questions, 'passage: ' for documents."""
    vecs = get_model().encode(
        [prefix + t for t in texts],
        normalize_embeddings=True,   # нормируем → косинус == dot
        batch_size=32,
    )
    return np.asarray(vecs, dtype=np.float32)
