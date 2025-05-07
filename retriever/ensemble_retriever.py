from dotenv import load_dotenv
from langchain_core.runnables import ConfigurableField
from langchain_teddynote import logging
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

"""
EnsembleRetriever는 여러 검색기를 결합하여 더 강력한 검색 결과를 제공하는 LangChain의 기능입니다. 이 검색기는 다양한 검색 알고리즘의 장점을 활용하여 단일 알고리즘보다 더 나은 성능을 달성할 수 있습니다.

주요 특징 
1. 여러 검색기 통합: 다양한 유형의 검색기를 입력으로 받아 결과를 결합합니다. 
2. 결과 재순위화: Reciprocal Rank Fusion 알고리즘을 사용하여 결과의 순위를 조정합니다. 
3. 하이브리드 검색: 주로 sparse retriever(예: BM25)와 dense retriever(예: 임베딩 유사도)를 결합하여 사용합니다.
"""

load_dotenv()
logging.langsmith("CH11-Retriever")

doc_list = [
    "I like apples",
    "I like apple company",
    "I like apple's iphone",
    "Apple is my favorite company",
    "I like apple's ipad",
    "I like apple's macbook",
]

bm25_retriever = BM25Retriever.from_texts(
    doc_list,
)
bm25_retriever.k = 1

embedding = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding,
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever]
).configurable_fields(
    weights=ConfigurableField(
        id="ensemble_weights",  # 검색 매개변수의 고유 식별자를 설정합니다.
        name="Ensemble Weights",    # 검색 매개변수의 이름을 설정합니다.
        description="Ensemble Weights", # 검색 매개변수에 대한 설명을 작성합니다.
    )
)

query = "my favorite fruit is apple"

config = {"configurable": {"ensemble_weights": [1, 0]}}
ensemble_result = ensemble_retriever.invoke(query, config=config)
bm25_result = bm25_retriever.invoke(query)
faiss_result = faiss_retriever.invoke(query)

print("[Ensemble Retriever]")
for doc in ensemble_result:
    print(f"Content: {doc.page_content}")
    print()

print("[BM25 Retriever]")
for doc in bm25_result:
    print(f"Content: {doc.page_content}")
    print()

print("[FAISS Retriever]")
for doc in faiss_result:
    print(f"Content: {doc.page_content}")
    print()
