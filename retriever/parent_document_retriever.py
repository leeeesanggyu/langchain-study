from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever

"""
ParentDocumentRetriever라는 도구가 사용됩니다. 이 도구는 문서를 작은 조각으로 나누고, 이 조각들을 관리합니다.
검색을 진행할 때는, 먼저 이 작은 조각들을 찾아낸 다음, 이 조각들이 속한 원본 문서(또는 더 큰 조각)의 식별자(ID)를 통해 전체적인 맥락을 파악할 수 있습니다.

여기서 '부모 문서'란, 작은 조각이 나누어진 원본 문서를 말합니다.
이는 전체 문서일 수도 있고, 비교적 큰 다른 조각일 수도 있습니다.
이 방식을 통해 문서의 의미를 정확하게 파악하면서도, 전체적인 맥락을 유지할 수 있게 됩니다.
"""

load_dotenv()
logging.langsmith("CH11-Retriever")

loaders = [
    TextLoader("./data/nlp-keywords.txt"),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# DB를 생성합니다.
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# 문서를 검색기에 추가합니다. docs는 문서 목록이고, ids는 문서의 고유 식별자 목록입니다.
retriever.add_documents(docs, ids=None, add_to_docstore=True)

# 저장소의 모든 키를 리스트로 반환합니다.
print(list(store.yield_keys()))

# 유사도 검색을 수행합니다.
sub_docs = vectorstore.similarity_search("Word2Vec")
print(sub_docs[0].page_content)

# 문서를 검색하여 가져옵니다.
retrieved_docs = retriever.invoke("Word2Vec")
# 검색된 문서의 문서의 페이지 내용의 길이를 출력합니다.
print(
    f"문서의 길이: {len(retrieved_docs[0].page_content)}",
    end="\n\n=====================\n\n",
)

# 문서의 일부를 출력합니다.
print(retrieved_docs[0].page_content[2000:2500])