import logging
from typing import List

from langchain.tools import tool
from ddgs import DDGS

from utils.rag import semantic_search_by_file

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """
    Searches the web for real-time information using DDGS (DuckDuckGo Search).
    Returns top results including title, snippet, and source URL.
    """
    logger.info(f"Executing web search for query: '{query}'")
    results: List[str] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                logger.debug(f"Web search hit: Title='{r['title']}', URL={r['href']}")
                results.append(f"Title: {r['title']}\nContent: {r['body']}\nSource: {r['href']}")
        if not results:
            logger.info("No results found for the query.")
            return "No results found for the query."
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"An error occurred during web search: {e}", exc_info=True)
        return f"An error occurred during web search: {str(e)}"


@tool
def semantic_search_chroma(query: str, user_id: str, file_id: str) -> str:
    """
    Retrieve relevant snippets from the vector store for the given user_id and file_id.
    This tool is always file-scoped; pass the file_id returned by /upload.
    """
    try:
        chunks = semantic_search_by_file(query, user_id=user_id, file_id=file_id, top_k=5)
        if not chunks:
            return "No relevant documents found."
        return "\n\n".join(chunks)
    except Exception as e:
        logger.error(f"Error in semantic_search_chroma: {e}", exc_info=True)
        return f"Error during semantic search: {str(e)}"


tools = [web_search, semantic_search_chroma]

tool_map = {t.name: t for t in tools}