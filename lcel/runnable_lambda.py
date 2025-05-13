from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter

load_dotenv()


def length_function(text):  # 텍스트의 길이를 반환하는 함수
    return len(text)


def _multiple_length_function(text1, text2):  # 두 텍스트의 길이를 곱하는 함수
    return len(text1) * len(text2)


def multiple_length_function(  # 2개 인자를 받는 함수로 연결하는 wrapper 함수
    _dict,
):  # 딕셔너리에서 "text1"과 "text2"의 길이를 곱하는 함수
    return _multiple_length_function(_dict["text1"], _dict["text2"])


# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template("what is {a} + {b}?")
# ChatOpenAI 모델 초기화
model = ChatOpenAI()

# 프롬프트와 모델을 연결하여 체인 생성
chain1 = prompt | model

# 체인 구성
chain = (
    {
        "a": itemgetter("input_1") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("input_1"), "text2": itemgetter("input_2")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
    | StrOutputParser()
)

# 주어진 인자들로 체인을 실행합니다.
print(chain.invoke({"input_1": "bar", "input_2": "gah"}))
