from dotenv import load_dotenv
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_response

# 예제가 많은 경우 프롬프트에 포함할 예제를 선택해야 할 수도 있습니다. Example Selector 는 이 작업을 담당하는 클래스입니다.
load_dotenv()
logging.langsmith("example-selector")

# Vector DB 생성 (저장소 이름, 임베딩 클래스)
chroma = Chroma("example_selector", OpenAIEmbeddings())

examples = [
    {
        "question": "스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
        추가 질문: 스티브 잡스는 몇 살에 사망했나요?
        중간 답변: 스티브 잡스는 56세에 사망했습니다.
        추가 질문: 아인슈타인은 몇 살에 사망했나요?
        중간 답변: 아인슈타인은 76세에 사망했습니다.
        최종 답변은: 아인슈타인
        """,
    },
    {
        "question": "네이버의 창립자는 언제 태어났나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
        추가 질문: 네이버의 창립자는 누구인가요?
        중간 답변: 네이버는 이해진에 의해 창립되었습니다.
        추가 질문: 이해진은 언제 태어났나요?
        중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
        최종 답변은: 1967년 6월 22일
        """,
    },
    {
        "question": "율곡 이이의 어머니가 태어난 해의 통치하던 왕은 누구인가요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
        추가 질문: 율곡 이이의 어머니는 누구인가요?
        중간 답변: 율곡 이이의 어머니는 신사임당입니다.
        추가 질문: 신사임당은 언제 태어났나요?
        중간 답변: 신사임당은 1504년에 태어났습니다.
        추가 질문: 1504년에 조선을 통치한 왕은 누구인가요?
        중간 답변: 1504년에 조선을 통치한 왕은 연산군입니다.
        최종 답변은: 연산군
        """,
    },
    {
        "question": "올드보이와 기생충의 감독이 같은 나라 출신인가요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
        추가 질문: 올드보이의 감독은 누구인가요?
        중간 답변: 올드보이의 감독은 박찬욱입니다.
        추가 질문: 박찬욱은 어느 나라 출신인가요?
        중간 답변: 박찬욱은 대한민국 출신입니다.
        추가 질문: 기생충의 감독은 누구인가요?
        중간 답변: 기생충의 감독은 봉준호입니다.
        추가 질문: 봉준호는 어느 나라 출신인가요?
        중간 답변: 봉준호는 대한민국 출신입니다.
        최종 답변은: 예
        """,
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,   # 여기에는 선택 가능한 예시 목록이 있습니다.
    OpenAIEmbeddings(), # 여기에는 의미적 유사성을 측정하는 데 사용되는 임베딩을 생성하는 임베딩 클래스가 있습니다.
    Chroma, # 여기에는 임베딩을 저장하고 유사성 검색을 수행하는 데 사용되는 VectorStore 클래스가 있습니다.
    k=1,    # 이것은 생성할 예시의 수입니다.
)

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

llm = ChatOpenAI(temperature=0)

chain = prompt | llm

stream_response(chain.stream({"question": "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"}))

