from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from langchain_teddynote.document_compressors import LLMChainExtractor
from langchain_text_splitters import CharacterTextSplitter

"""
DocumentCompressorPipeline 을 사용하면 여러 compressor를 순차적으로 결합할 수 있습니다.
Compressor와 함께 BaseDocumentTransformer를 파이프라인에 추가할 수 있는데, 이는 맥락적 압축을 수행하지 않고 단순히 문서 집합에 대한 변환을 수행합니다.
예를 들어, TextSplitter는 문서를 더 작은 조각으로 분할하기 위해 document transformer로 사용될 수 있으며, EmbeddingsRedundantFilter는 문서 간의 임베딩 유사성(기본값: 0.95 유사도 이상을 중복 문서로 간주) 을 기반으로 중복 문서를 필터링하는 데 사용될 수 있습니다.
"""

load_dotenv()
logging.langsmith("CH11-Retriever")

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"문서 {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

embeddings = OpenAIEmbeddings()

loader = TextLoader("./data/nlp-keywords.txt")
text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=30, chunk_overlap=0
)
texts = loader.load_and_split(text_splitter)

retriever = FAISS.from_documents(texts, embeddings).as_retriever()

pipeline_compressor = DocumentCompressorPipeline(
    # 문서 압축 파이프라인을 생성하고, 분할기, 중복 필터, 관련성 필터, LLMChainExtractor를 변환기로 설정합니다.
    transformers=[
        CharacterTextSplitter(chunk_size=300, chunk_overlap=0), # 문자 기반 텍스트 분할기를 생성하고, 청크 크기를 300으로, 청크 간 중복을 0으로 설정합니다.
        EmbeddingsRedundantFilter(embeddings=embeddings),   # 임베딩을 사용하여 중복 필터를 생성합니다.
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.86), # 임베딩을 사용하여 관련성 필터를 생성하고, 유사도 임계값을 0.86으로 설정합니다.
        LLMChainExtractor.from_llm(ChatOpenAI(temperature=0, model="gpt-4o-mini")),
    ]
)

compression_retriever = ContextualCompressionRetriever(
    # 기본 압축기로 pipeline_compressor를 사용하고, 기본 검색기로 retriever를 사용하여 ContextualCompressionRetriever를 초기화합니다.
    base_compressor=pipeline_compressor,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(
    "Semantic Search 에 대해서 알려줘."
)
pretty_print_docs(compressed_docs)
