"""One-search adaptive centroid strategy selected in the isolated retrieval lab."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from .experiment import unique
from .vector_search import DenseVectorSearch, l2_normalize


def choose_question_mass(question: str, subqueries: Iterable[str],
                        threshold_chars: int = 39) -> Tuple[float, List[str], int]:
    question = question.strip()
    parts = [part for part in unique(subqueries) if part != question]
    char_count = sum(len(part) for part in parts)
    question_mass = 0.60 if char_count <= threshold_chars else 0.32
    return question_mass, parts, char_count


def compose_centroid(question_vector: Sequence[float],
                     subquery_vectors: Sequence[Sequence[float]],
                     question_mass: float) -> List[float]:
    question = l2_normalize(question_vector)
    parts = [l2_normalize(vector) for vector in subquery_vectors]
    if not parts:
        return question
    part_mass = (1.0 - question_mass) / len(parts)
    result = [
        question_mass * question[index]
        + sum(part_mass * vector[index] for vector in parts)
        for index in range(len(question))
    ]
    return l2_normalize(result)


def adaptive_search(searcher: DenseVectorSearch, question: str,
                    subqueries: Iterable[str], spot_names: List[str],
                    candidate_pool: int = 20,
                    threshold_chars: int = 39) -> Dict[str, Any]:
    question = question.strip()
    question_mass, parts, char_count = choose_question_mass(
        question, subqueries, threshold_chars
    )
    texts = [question, *parts]
    encoded = searcher.encode(texts)
    vector = compose_centroid(encoded[0], encoded[1:], question_mass)
    hits = searcher.search_vector(vector, candidate_pool)
    hits = [hit for hit in hits if searcher.matches_spot(hit["payload"], spot_names)]
    return {
        "hits": hits,
        "question_mass": question_mass,
        "subquery_char_count": char_count,
        "subquery_count": len(parts),
    }
