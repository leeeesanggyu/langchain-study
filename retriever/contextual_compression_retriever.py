from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_teddynote.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

"""
검색된 문서를 그대로 즉시 반환하는 대신, 주어진 질의의 맥락을 사용하여 문서를 압축함으로써 관련 정보만 반환되도록 할 수 있습니다.
여기서 "압축"은 개별 문서의 내용을 압축하는 것과 문서를 전체적으로 필터링하는 것 모두를 의미합니다.
ContextualCompressionRetriever 는 질의를 base retriever에 전달하고, 초기 문서를 가져와 Document Compressor를 통과시킵니다.
Document Compressor는 문서 목록을 가져와 문서의 내용을 줄이거나 문서를 완전히 삭제하여 목록을 축소합니다.
"""

load_dotenv()
logging.langsmith("CH11-Retriever")

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"문서 {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

loader = TextLoader("./data/nlp-keywords.txt")
text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=30, chunk_overlap=0
)
texts = loader.load_and_split(text_splitter)

retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  # OpenAI 언어 모델 초기화

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever( # 문서 압축기와 리트리버를 사용하여 컨텍스트 압축 리트리버 생성
    base_compressor=compressor,
    base_retriever=retriever,
)

compressed_docs = (
    compression_retriever.invoke(  # 컨텍스트 압축 리트리버를 사용하여 관련 문서 검색
        "Semantic Search 에 대해서 알려줘."
    )
)
pretty_print_docs(compressed_docs)  # 검색된 문서를 예쁘게 출력