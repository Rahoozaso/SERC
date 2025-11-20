import logging
import os
from typing import Dict, Any, List
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, config: Dict[str, Any]):
        if not os.environ.get("TAVILY_API_KEY"):
            logger.warning("[RAGRetriever] TAVILY_API_KEY not set.")
        
        rag_config = config.get('rag_config', {})
        self.top_k = rag_config.get('rag_top_k_results', 10) 
        self.search_depth = rag_config.get('search_depth', "advanced") 
        self.max_ctx_chars = rag_config.get('max_context_characters', 100000) # 20k로 넉넉하게

        try:
            self.retriever = TavilySearchAPIRetriever(
                k=self.top_k,
                search_depth=self.search_depth,
                include_raw_content=False,
                include_images=False
            )
            logger.info(f"[RAGRetriever] Init (k={self.top_k}, depth={self.search_depth})")
        
        except Exception as e:
            logger.error(f"[RAGRetriever] Init Failed: {e}")
            raise e

    def retrieve(self, query: str) -> str:
        clean_query = query.strip()
        if not clean_query:
            return "[No query provided]"

        logger.info(f"    [RAG Retrieve] Query: '{clean_query[:50]}...'")
        
        try:
            documents: List[Document] = self.retriever.invoke(clean_query)
            
            if not documents:
                return "[No relevant documents found]"

            formatted_results = []
            current_length = 0
            
            for i, doc in enumerate(documents):
                content = doc.page_content.strip()
                source = doc.metadata.get('source', 'Unknown')
                
                if len(content) < 30: continue

                snippet = f"[Source {i+1}]: {source}\n{content}"
                
                if current_length + len(snippet) > self.max_ctx_chars:
                    break
                
                formatted_results.append(snippet)
                current_length += len(snippet)

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"[RAG Retrieve] Error: {e}")
            return "[Retrieval error]"