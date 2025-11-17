import logging
import os
from typing import Dict, Any, List
# [수정] TavilyRetriever를 임포트합니다.
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, config: Dict[str, Any]):
        if not os.environ.get("TAVILY_API_KEY"):
            logger.warning("[RAGRetriever] TAVILY_API_KEY 환경 변수가 설정되지 않았습니다. 검색이 실패할 수 있습니다.")
            # API 키가 없으면 에러를 발생시킬 수도 있습니다.
            # raise ValueError("TAVILY_API_KEY가 필요합니다.")
        
        rag_config = config.get('rag_config', {})
        self.top_k = rag_config.get('rag_top_k_results', 8) 

        try:
            # [수정] WikipediaRetriever 대신 TavilySearchAPIRetriever를 사용합니다.
            self.retriever = TavilySearchAPIRetriever(k=self.top_k)
            logger.info(f"[RAGRetriever] Tavily AI (Live Web Search) (k={self.top_k})로 초기화됨.")
        
        except Exception as e:
            logger.error(f"[RAGRetriever] TavilySearchAPIRetriever 초기화 실패: {e}")
            raise e

    def retrieve(self, query: str) -> str:
        """
        [수정] Tavily AI로 '시맨틱 웹 검색'을 수행합니다.
        """
        logger.info(f"    [RAG Retrieve] Tavily AI 쿼리: '{query[:50]}...'")
        
        try:
            # Tavily는 의미론적으로 가장 유사한 '스니펫'을 검색
            documents: List[Document] = self.retriever.invoke(query)
            
            if not documents:
                logger.warning(f"    [RAG Retrieve] Tavily가 문서를 찾지 못함: '{query}'")
                return "[No relevant documents found]"

            # 검색된 모든 문서의 내용을 하나의 문자열로 결합
            context_string = "\n---\n".join([doc.page_content for doc in documents])
            return context_string

        except Exception as e:
            logger.error(f"[RAG Retrieve] Tavily 검색 중 오류 발생: {e}")
            return "[Retrieval error]"