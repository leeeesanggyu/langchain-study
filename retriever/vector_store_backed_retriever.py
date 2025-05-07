from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

"""
VectorStore 지원 검색기 는 vector store를 사용하여 문서를 검색하는 retriever입니다.

Vector store에 구현된 유사도 검색(similarity search) 이나 MMR 과 같은 검색 메서드를 사용하여 vector store 내의 텍스트를 쿼리합니다.
"""

load_dotenv()
logging.langsmith("CH11-Retriever")

loader = TextLoader("./data/nlp-keywords.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

for i, doc in enumerate(split_docs):
    print(f"--- chunk {i} ---\n{doc.page_content}\n")

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(split_docs, embeddings)

retriever = db.as_retriever()
config = {
    "configurable": {
        "search_type": "mmr",
        "search_kwargs": {"k": 2, "fetch_k": 10, "lambda_mult": 0.6},
    }
}

docs = retriever.invoke("Word2Vec 은 무엇인가요?", config=config)

for doc in docs:
    print(f"===== {doc.page_content}")
