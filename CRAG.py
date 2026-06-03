
# Import everything from the engine for backwards compatibility
from crag_engine import (
    embedding_model,
    reranker,
    all_chunks,
    bm25,
    vector_db,
    splitter,
    llm,
    load_vector_db,
    save_chunks,
    load_chunks,
    rebuild_bm25,
    add_pdf,
    hybrid_retrieval,
    rerank_documents,
    generate_answer,
    CRAG,
    query_with_scores,
    initialize_crag
)

__all__ = [
    'embedding_model',
    'reranker',
    'all_chunks',
    'bm25',
    'vector_db',
    'splitter',
    'llm',
    'load_vector_db',
    'save_chunks',
    'load_chunks',
    'rebuild_bm25',
    'add_pdf',
    'hybrid_retrieval',
    'rerank_documents',
    'generate_answer',
    'CRAG',
    'query_with_scores',
    'initialize_crag'
]
