from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from unittest.mock import patch

import httpx
from openai import RateLimitError

"""
LLM 애플리케이션에는 LLM API 문제, 모델 출력 품질 저하, 다른 통합 관련 이슈 등 다양한 오류/실패가 존재합니다.
이러한 문제를 우아하게 처리하고 격리하는데 fallback 기능을 활용할 수 있습니다.
중요한 점은 fallback 을 LLM 수준뿐만 아니라 전체 실행 가능한 수준에 적용할 수 있다는 것입니다.

기본적으로 많은 LLM 래퍼(wrapper)는 오류를 포착하고 재시도합니다.
fallback 을 사용할 때는 이러한 기본 동작을 해제하는 것이 좋습니다.
그렇지 않으면 첫 번째 래퍼가 계속 재시도하고 실패하지 않을 것입니다.
"""

load_dotenv()

request = httpx.Request("GET", "/")  # GET 요청을 생성합니다.
response = httpx.Response(200, request=request)  # 200 상태 코드와 함께 응답을 생성합니다.
# "rate limit" 메시지와 응답 및 빈 본문을 포함하는 RateLimitError를 생성합니다.
error = RateLimitError("rate limit", response=response, body="")

# max_retries를 0으로 설정하여 속도 제한 등으로 인한 재시도를 방지합니다.
openai_llm = ChatOpenAI(max_retries=0)
anthropic_llm = ChatAnthropic(model="claude-3-opus-20240229")

# openai_llm을 기본으로 사용하고, 실패 시 anthropic_llm을 대체로 사용하도록 설정합니다.
llm = openai_llm.with_fallbacks([anthropic_llm])

# OpenAI LLM을 먼저 사용하여 오류가 발생하는 것을 보여줍니다.
# with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
#     try:
#         # "닭이 길을 건넌 이유는 무엇일까요?"라는 질문을 OpenAI LLM에 전달합니다.
#         print(openai_llm.invoke("Why did the chicken cross the road?"))
#     except RateLimitError:
#         # 오류가 발생하면 오류를 출력합니다.
#         print("에러 발생")

# OpenAI API 호출 시 에러가 발생하는 경우 Anthropic 으로 대체하는 코드
with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
    try:
        # "대한민국의 수도는 어디야?"라는 질문을 언어 모델에 전달하여 응답을 출력합니다.
        print(llm.invoke("대한민국의 수도는 어디야?"))
    except RateLimitError:
        # RateLimitError가 발생하면 "에러 발생"를 출력합니다.
        print("에러 발생")
