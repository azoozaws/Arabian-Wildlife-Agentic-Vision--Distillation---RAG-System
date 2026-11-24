"""Dataset merging, RAG document storage, Qwen3-Embedding ChromaDB persistence, and natural-language RAG query engine."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "clean_orphan_labels": ("app.data.dataset_builder", "clean_orphan_labels"),
    "copy_pair": ("app.data.dataset_builder", "copy_pair"),
    "build_merged_dataset": ("app.data.dataset_builder", "build_merged_dataset"),
    "build_rag_document": ("app.data.rag_store", "build_rag_document"),
    "save_rag_documents_jsonl": ("app.data.rag_store", "save_rag_documents_jsonl"),
    "load_rag_documents_jsonl": ("app.data.rag_store", "load_rag_documents_jsonl"),
    "filter_rag_documents": ("app.data.rag_store", "filter_rag_documents"),
    "summarize_rag_corpus": ("app.data.rag_store", "summarize_rag_corpus"),
    "export_vector_db_payloads": ("app.data.rag_store", "export_vector_db_payloads"),
    "QwenEmbeddingFunction": ("app.data.rag_store", "QwenEmbeddingFunction"),
    "get_qwen_embedding_function": ("app.data.rag_store", "get_qwen_embedding_function"),
    "last_token_pool": ("app.data.rag_store", "last_token_pool"),
    "upsert_rag_documents_to_chromadb": ("app.data.rag_store", "upsert_rag_documents_to_chromadb"),
    "rebuild_chromadb_from_jsonl": ("app.data.rag_store", "rebuild_chromadb_from_jsonl"),
    "query_chromadb_rag": ("app.data.rag_store", "query_chromadb_rag"),
    "ParsedRAGQuery": ("app.data.rag_query_engine", "ParsedRAGQuery"),
    "normalize_arabic_text": ("app.data.rag_query_engine", "normalize_arabic_text"),
    "parse_natural_rag_query": ("app.data.rag_query_engine", "parse_natural_rag_query"),
    "resolve_local_media_path": ("app.data.rag_query_engine", "resolve_local_media_path"),
    "search_wildlife_rag": ("app.data.rag_query_engine", "search_wildlife_rag"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazily resolve package exports to avoid runpy pre-import warnings on `python -m app.data.rag_store`."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    return getattr(import_module(mod_name), attr_name)
