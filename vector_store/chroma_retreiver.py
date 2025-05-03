from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
logging.langsmith("CH10-VectorStores")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=0)

loader1 = TextLoader("data/nlp-keywords.txt")
split_doc1 = loader1.load_and_split(text_splitter)
print(len(split_doc1))

# for doc in split_doc1:
#     print(f"==== {doc.page_content}")

loader2 = TextLoader("data/finance-keywords.txt")
split_doc2 = loader2.load_and_split(text_splitter)
print(len(split_doc2))

DB_PATH = "./chroma_db"

Chroma.from_documents(
    split_doc1, OpenAIEmbeddings(), persist_directory=DB_PATH, collection_name="my_db"
)
Chroma.from_documents(
    split_doc2, OpenAIEmbeddings(), persist_directory=DB_PATH, collection_name="my_db2"
)

db = Chroma.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=OpenAIEmbeddings(),
    collection_name="nlp",
)

"""
"mmr"은 Maximal Marginal Relevance 방식으로 중복이 적고 다양성이 높은 결과를 반환합니다.
"similarity"는 단순 코사인 유사도 기반 상위 k개 문서를 반환합니다.

fetch_k: 유사도 기반으로 우선 fetch_k 개수를 먼저 가져온 다음, 그 중에서 다양성을 고려하여 k개를 고릅니다.
lambda_mult: 다양성과 유사도 사이의 가중치 비율을 설정합니다. 0.0이면 다양성만, 1.0이면 유사도만을 고려합니다.
"""
# retriever = db.as_retriever()
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.25, "fetch_k": 10, "score_threshold": 0.9}
)
retrieved = retriever.invoke("dividend yield가 뭐야?")

for doc in retrieved:
    print(f"==== {doc}")

