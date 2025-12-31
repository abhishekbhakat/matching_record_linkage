"""ModernBERT-Embed semantic matching using Nomic AI's embedding model."""

import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "nomic-ai/modernbert-embed-base"
MAX_LENGTH = 512

_model = None
_tokenizer = None


def _get_model():
    """Lazy load model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model.eval()
    return _model, _tokenizer


def _mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode_texts(texts: list[str]) -> torch.Tensor:
    """Encode texts into embeddings."""
    model, tokenizer = _get_model()
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    embeddings = _mean_pooling(outputs, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def cosine_similarity_matrix(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix."""
    return torch.mm(emb1, emb2.t())


def modernbert_match(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.8,
) -> pl.DataFrame:
    """Semantic matching using ModernBERT embeddings.

    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame
        fields: List of column names to match on
        weights: Dict of {field: weight} for weighted scoring
        threshold: Minimum cosine similarity [0..1]

    Returns:
        DataFrame with columns: left_idx, right_idx, score
    """
    total_weight = sum(weights.get(f, 1.0) for f in fields)
    if total_weight == 0:
        total_weight = float(len(fields))

    left_texts = []
    right_texts = []
    
    for row in df_left.iter_rows(named=True):
        text = " ".join(str(row.get(f, "") or "") for f in fields)
        left_texts.append(text)
    
    for row in df_right.iter_rows(named=True):
        text = " ".join(str(row.get(f, "") or "") for f in fields)
        right_texts.append(text)

    left_emb = encode_texts(left_texts)
    right_emb = encode_texts(right_texts)
    
    sim_matrix = cosine_similarity_matrix(left_emb, right_emb)

    matches = []
    for i in range(len(df_left)):
        for j in range(len(df_right)):
            score = sim_matrix[i, j].item()
            if score >= threshold:
                matches.append({
                    "left_idx": i,
                    "right_idx": j,
                    "score": round(score, 4),
                })

    if not matches:
        return pl.DataFrame({
            "left_idx": pl.Series([], dtype=pl.Int64),
            "right_idx": pl.Series([], dtype=pl.Int64),
            "score": pl.Series([], dtype=pl.Float64),
        })

    return pl.DataFrame(matches)
