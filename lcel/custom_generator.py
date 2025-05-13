from typing import Iterator, List

from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

"""
제너레이터 함수(즉, yield 키워드를 사용하고 이터레이터처럼 동작하는 함수)를 LCEL 파이프라인에서 사용할 수 있습니다.
이러한 제너레이터의 시그니처는 Iterator[Input] -> Iterator[Output]이어야 합니다. 
비동기 제너레이터의 경우에는 AsyncIterator[Input] -> AsyncIterator[Output]입니다.

- 사용자 정의 출력 파서 구현
- 이전 단계의 출력을 수정하면서 스트리밍 기능 유지
"""

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    # 주어진 회사와 유사한 5개의 회사를 쉼표로 구분된 목록으로 작성하세요.
    "Write a comma-separated list of 5 companies similar to: {company}"
)
# 온도를 0.0으로 설정하여 ChatOpenAI 모델을 초기화합니다.
model = ChatOpenAI(temperature=0.0, model="gpt-4-turbo-preview")

# 프롬프트와 모델을 연결하고 문자열 출력 파서를 적용하여 체인을 생성합니다.
str_chain = prompt | model | StrOutputParser()


# 입력으로 llm 토큰의 반복자를 받아 쉼표로 구분된 문자열 리스트로 분할하는 사용자 정의 파서입니다.
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    # 쉼표가 나올 때까지 부분 입력을 보관합니다.
    buffer = ""
    for chunk in input:
        # 현재 청크를 버퍼에 추가합니다.
        buffer += chunk
        # 버퍼에 쉼표가 있는 동안 반복합니다.
        while "," in buffer:
            # 버퍼를 쉼표로 분할합니다.
            comma_index = buffer.index(",")
            # 쉼표 이전의 모든 내용을 반환합니다.
            yield [buffer[:comma_index].strip()]
            # 나머지는 다음 반복을 위해 저장합니다.
            buffer = buffer[comma_index + 1 :]
    # 마지막 청크를 반환합니다.
    yield [buffer.strip()]


list_chain = str_chain | split_into_list  # 문자열 체인을 리스트로 분할합니다.

# 생성한 list_chain 이 문제없이 스트리밍되는지 확인합니다.
for chunk in list_chain.stream({"company": "Google"}):
    print(chunk, flush=True)  # 각 청크를 출력하고, 버퍼를 즉시 플러시합니다.

# list_chain 에 데이터를 invoke 합니다.
print(list_chain.invoke({"company": "Google"}))
