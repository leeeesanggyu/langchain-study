from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

load_dotenv()


class Quiz(BaseModel):
    """4지선다형 퀴즈의 정보를 추출합니다"""

    question: str = Field(..., description="퀴즈의 질문")
    level: str = Field(..., description="퀴즈의 난이도를 나타냅니다. (쉬움, 보통, 어려움)")
    options: List[str] = Field(..., description="퀴즈의 4개의 선택지 입니다.")


llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a world-famous quizzer and generates quizzes in structured formats.",
        ),
        (
            "system",
            "You must answer in Korean.",
        ),
        (
            "human",
            "TOPIC 에 제시된 내용과 관련한 4지선다형 퀴즈를 출제해 주세요. 만약, 실제 출제된 기출문제가 있다면 비슷한 문제를 만들어 출제하세요."
            "단, 문제에 TOPIC 에 대한 내용이나 정보는 포함하지 마세요. \nTOPIC:\n{topic}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

# 구조화된 출력을 위한 모델 생성
llm_with_structured_output = llm.with_structured_output(Quiz)

# 퀴즈 생성 체인 생성
chain = prompt | llm_with_structured_output

# 퀴즈 생성을 요청합니다.
generated_quiz = chain.invoke({"topic": "Spring Framework"})

# 생성된 퀴즈 출력
print(f"{generated_quiz.question} (난이도: {generated_quiz.level})\n")
for i, opt in enumerate(generated_quiz.options):
    print(f"{i + 1}) {opt}")
