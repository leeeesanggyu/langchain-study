import time

from dotenv import load_dotenv
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_text_splitters import CharacterTextSplitter

"""
Embeddings를 캐싱하는 것은 CacheBackedEmbeddings를 사용하여 수행될 수 있습니다. 캐시 지원 embedder는 embeddings를 키-값 저장소에 캐싱하는 embedder 주변에 래퍼입니다. 텍스트는 해시되고 이 해시는 캐시에서 키로 사용됩니다.

CacheBackedEmbeddings를 초기화하는 주요 지원 방법은 from_bytes_store입니다. 이는 다음 매개변수를 받습니다:

underlying_embeddings: 임베딩을 위해 사용되는 embedder.
document_embedding_cache: 문서 임베딩을 캐싱하기 위한 ByteStore 중 하나.
namespace: (선택 사항, 기본값은 "") 문서 캐시를 위해 사용되는 네임스페이스. 이 네임스페이스는 다른 캐시와의 충돌을 피하기 위해 사용됩니다. 예를 들어, 사용된 임베딩 모델의 이름으로 설정하세요.
주의: 동일한 텍스트가 다른 임베딩 모델을 사용하여 임베딩될 때 충돌을 피하기 위해 namespace 매개변수를 설정하는 것이 중요합니다.
"""

load_dotenv()
embedding = OpenAIEmbeddings()
store = LocalFileStore("./cache/")

# 캐시를 지원하는 임베딩 생성
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
)

raw_documents = TextLoader("./example.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

start = time.time()
db = FAISS.from_documents(documents, cached_embedder)
end = time.time()
print(f"실행 시간: {end - start:.6f}초")

start2 = time.time()
db2 = FAISS.from_documents(documents, cached_embedder)
end2 = time.time()
print(f"실행 시간: {end2 - start2:.6f}초")