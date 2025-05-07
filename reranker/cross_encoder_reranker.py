from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

"""
Cross encoder reranker는 검색 증강 생성(RAG) 시스템의 성능을 향상시키기 위해 사용되는 기술입니다.
이 문서는 Hugging Face의 cross encoder 모델을 사용하여 retriever에서 reranker를 구현하는 방법을 설명합니다.

- 일반적으로 초기 검색에서 상위 k개의 문서에 대해서만 reranking 수행
- Bi-encoder로 빠르게 후보를 추출한 후, Cross encoder로 정확도를 높이는 방식으로 활용
"""

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

documents = TextLoader("./data/appendix-keywords.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddingsModel = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-dot-v5"
)

retriever = FAISS.from_documents(texts, embeddingsModel).as_retriever(
    search_kwargs={"k": 10}
)

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

query = "Word2Vec 에 대해서 알려줄래?"

docs = compression_retriever.invoke(query)
pretty_print_docs(docs)





# 문서 압축 검색기 초기화


# 압축된 문서 검색
compressed_docs = compression_retriever.invoke("Word2Vec 에 대해서 알려줄래?")

# 문서 출력
pretty_print_docs(compressed_docs)

