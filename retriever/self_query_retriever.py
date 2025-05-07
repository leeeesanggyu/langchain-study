from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator

"""
SelfQueryRetriever 는 자체적으로 질문을 생성하고 해결할 수 있는 기능을 갖춘 검색 도구입니다.

이는 사용자가 제공한 자연어 질의를 바탕으로, query-constructing LLM chain을 사용해 구조화된 질의를 만듭니다.
그 후, 이 구조화된 질의를 기본 벡터 데이터 저장소(VectorStore)에 적용하여 검색을 수행합니다.

이 과정을 통해, SelfQueryRetriever 는 단순히 사용자의 입력 질의를 저장된 문서의 내용과 의미적으로 비교하는 것을 넘어서, 사용자의 질의에서 문서의 메타데이터에 대한 필터를 추출 하고, 이 필터를 실행하여 관련된 문서를 찾을 수 있습니다.
"""

load_dotenv()
logging.langsmith("CH11-Retriever")

# 화장품 상품의 설명과 메타데이터 생성
docs = [
    Document(
        page_content="수분 가득한 히알루론산 세럼으로 피부 속 깊은 곳까지 수분을 공급합니다.",
        metadata={"year": 2024, "category": "스킨케어", "user_rating": 4.7},
    ),
    Document(
        page_content="24시간 지속되는 매트한 피니시의 파운데이션, 모공을 커버하고 자연스러운 피부 표현이 가능합니다.",
        metadata={"year": 2023, "category": "메이크업", "user_rating": 4.5},
    ),
    Document(
        page_content="식물성 성분으로 만든 저자극 클렌징 오일, 메이크업과 노폐물을 부드럽게 제거합니다.",
        metadata={"year": 2023, "category": "클렌징", "user_rating": 4.8},
    ),
    Document(
        page_content="비타민 C 함유 브라이트닝 크림, 칙칙한 피부톤을 환하게 밝혀줍니다.",
        metadata={"year": 2023, "category": "스킨케어", "user_rating": 4.6},
    ),
    Document(
        page_content="롱래스팅 립스틱, 선명한 발색과 촉촉한 사용감으로 하루종일 편안하게 사용 가능합니다.",
        metadata={"year": 2024, "category": "메이크업", "user_rating": 4.4},
    ),
    Document(
        page_content="자외선 차단 기능이 있는 톤업 선크림, SPF50+/PA++++ 높은 자외선 차단 지수로 피부를 보호합니다.",
        metadata={"year": 2024, "category": "선케어", "user_rating": 4.9},
    ),
]

# 벡터 저장소 생성
vectorstore = Chroma.from_documents(
    docs, OpenAIEmbeddings(model="text-embedding-3-small")
)

# 메타데이터 필드 정보 생성
metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The category of the cosmetic product. One of ['스킨케어', '메이크업', '클렌징', '선케어']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the cosmetic product was released",
        type="integer",
    ),
    AttributeInfo(
        name="user_rating",
        description="A user rating for the cosmetic product, ranging from 1 to 5",
        type="float",
    ),
]

# 문서 내용 설명과 메타데이터 필드 정보를 사용하여 쿼리 생성기 프롬프트를 가져옵니다.
prompt = get_query_constructor_prompt(
    "Brief summary of a cosmetic product",  # 문서 내용 설명
    metadata_field_info,  # 메타데이터 필드 정보
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StructuredQueryOutputParser.from_components()

query_constructor = prompt | llm | output_parser

query = "2023년도에 출시한 상품 중 평점이 4.5 이상인 상품중에서 스킨케어 제품을 추천해주세요"

# 쿼리 생성기를 호출하여 주어진 질문에 대한 쿼리를 생성합니다.
query_output = query_constructor.invoke(
    {
        "query": query
    }
)
print(query_output.filter.arguments)

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
)
print(retriever.invoke(query))
