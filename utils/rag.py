import os
import uuid
import tempfile
import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from llama_parse import LlamaParse

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.getcwd(), "chroma_data"))
RAG_COLLECTION = os.getenv("RAG_COLLECTION", "documents_collection")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PARSE_KEY = os.getenv("PARSE_KEY")

_vectorstore: Optional[Chroma] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        # Default to CPU; users can override with model_kwargs via env if needed
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
        logger.info(f"Loaded HuggingFaceEmbeddings: {EMBEDDING_MODEL_NAME}")
    return _embeddings


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _vectorstore = Chroma(
            collection_name=RAG_COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        logger.info(f"Chroma vectorstore ready at '{CHROMA_DIR}' in collection '{RAG_COLLECTION}'")
    return _vectorstore


def store_file_embeddings(file_bytes: bytes, filename: str) -> dict:
    """
    Parse the uploaded file, produce one chunk per page, embed, and upsert to Chroma via LangChain.
    Each chunk metadata includes 'page_number' and 'source_file'.
    Returns summary metadata for logging/diagnostics.
    """
    if not PARSE_KEY:
        logger.error("PARSE_KEY environment variable not set.")
        raise EnvironmentError("PARSE_KEY environment variable not set.")

    _get_vectorstore()  # ensure initialized

    tmp_path = None
    try:
        # Persist file to a temporary path for LlamaParse
        file_ext = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        logger.debug(f"Temporary file for parsing created at {tmp_path}")

        parser = LlamaParse(api_key=PARSE_KEY, result_type="text", verbose=True)
        documents = parser.load_data([tmp_path])
        if not documents:
            logger.warning(f"No documents extracted from file '{filename}'")
            return {"filename": filename, "chunks": 0}

        ids: List[str] = []
        docs: List[Document] = []

        page_count = 0
        for idx, doc in enumerate(documents):
            try:
                text = getattr(doc, "text", "") or ""
                if not text.strip():
                    continue
                meta = getattr(doc, "metadata", None) or getattr(doc, "extra_info", {}) or {}
                page_number = int(meta.get("page") or meta.get("page_number") or (idx + 1))
                ids.append(str(uuid.uuid4()))
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_file": os.path.basename(filename),
                            "page_number": page_number,
                        },
                    )
                )
                page_count += 1
            except Exception as page_e:
                logger.warning(f"Skipping page {idx+1} due to error: {page_e}", exc_info=True)

        if not docs:
            logger.info(f"No non-empty pages to add for file '{filename}'")
            return {"filename": filename, "chunks": 0}

        vs = _get_vectorstore()
        vs.add_documents(documents=docs, ids=ids)
        logger.info(f"Ingested {len(docs)} page-chunks into Chroma from {filename}")
        return {"filename": filename, "chunks": len(docs), "pages": page_count}
    except Exception as e:
        logger.error(f"Error during RAG ingestion for file '{filename}': {e}", exc_info=True)
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"Temporary file deleted: {tmp_path}")


def semantic_search(query: str, top_k: int = 5) -> List[str]:
    """
    Perform a semantic search against Chroma and return top chunks of text.
    """
    try:
        vs = _get_vectorstore()
        docs = vs.similarity_search(query, k=top_k)
        if not docs:
            logger.info("Semantic search returned no results")
            return []
        chunks: List[str] = []
        for d in docs:
            meta = d.metadata or {}
            source_file = meta.get("source_file")
            page_info = meta.get("page_number")
            prefix = f"[source: {source_file} | page: {page_info}]\n" if source_file or page_info else ""
            chunks.append(prefix + (d.page_content or ""))
        logger.debug(f"Semantic search returned {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        return []
