import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# Init once (do this near your CONFIG area)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load the model once
_EMBED_MODEL = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

def embed_one(text: str,
              task_type: str = "RETRIEVAL_DOCUMENT",
              output_dim: int | None = None) -> np.ndarray:
    """
    Returns a 1D np.ndarray embedding for a single input text.
    Uses Vertex AI SDK (no batches). Set output_dim if you need a specific size.
    """
    inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
    if output_dim is not None:
        resp = _EMBED_MODEL.get_embeddings(inputs, output_dimensionality=int(output_dim))
    else:
        resp = _EMBED_MODEL.get_embeddings(inputs)
    return np.array(resp[0].values, dtype=np.float32)
