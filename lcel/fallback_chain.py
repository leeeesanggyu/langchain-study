from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from unittest.mock import patch

import httpx
from openai import RateLimitError
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

"""
LLM 애플리케이션에는 LLM API 문제, 모델 출력 품질 저하, 다른 통합 관련 이슈 등 다양한 오류/실패가 존재합니다.
이러한 문제를 우아하게 처리하고 격리하는데 fallback 기능을 활용할 수 있습니다.
중요한 점은 fallback 을 LLM 수준뿐만 아니라 전체 실행 가능한 수준에 적용할 수 있다는 것입니다.

기본적으로 많은 LLM 래퍼(wrapper)는 오류를 포착하고 재시도합니다.
fallback 을 사용할 때는 이러한 기본 동작을 해제하는 것이 좋습니다.
그렇지 않으면 첫 번째 래퍼가 계속 재시도하고 실패하지 않을 것입니다.
"""

load_dotenv()

prompt_template = (
    "질문에 짧고 간결하게 답변해 주세요.\n\nQuestion:\n{question}\n\nAnswer:"
)
prompt = PromptTemplate.from_template(prompt_template)

chat_model = ChatOpenAI(model_name="gpt-fake")
bad_chain = prompt | chat_model

fallback_chain1 = prompt | ChatOpenAI(model="gpt-3.6-turbo") # 오류
fallback_chain2 = prompt | ChatOpenAI(model="gpt-3.5-turbo") # 정상
fallback_chain3 = prompt | ChatOpenAI(model="gpt-4-turbo-preview") # 정상

chain = bad_chain.with_fallbacks([fallback_chain1, fallback_chain2, fallback_chain3])

print(chain.invoke({"question": "대한민국의 수도는 어디야?"}))
