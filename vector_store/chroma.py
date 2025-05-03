from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
logging.langsmith("CH10-VectorStores")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

loader1 = TextLoader("data/nlp-keywords.txt")
split_doc1 = loader1.load_and_split(text_splitter)
print(len(split_doc1))

DB_PATH = "./chroma_db"

save_persist_db = Chroma.from_documents(
    split_doc1, OpenAIEmbeddings(), persist_directory=DB_PATH, collection_name="my_db"
)

persist_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_db",
)
# print(persist_db.get())

search = persist_db.similarity_search(query="word2vec가 뭐야?", k=3)
print(f"before : {search}")

persist_db.add_documents(
    [
        Document(
            page_content="안녕하세요! 이번엔 도큐먼트를 새로 추가해 볼께요",
            metadata={"source": "mydata.txt"},
            id="1",
        )
    ]
)

search = persist_db.get("1")
print(f"after : {search}")

search = persist_db.similarity_search(query="도큐먼트 추가!", k=3)
print(f"after : {search}")
