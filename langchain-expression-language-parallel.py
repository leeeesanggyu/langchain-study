from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
logging.langsmith("langchain-expression-language")

model = ChatOpenAI(model_name="gpt-4-turbo")


chain1 = (
    PromptTemplate.from_template("{country} 의 수도는 어디야?")
    | model
    | StrOutputParser()
)

chain2 = (
    PromptTemplate.from_template("{country} 의 면적은 얼마야?")
    | model
    | StrOutputParser()
)

# 위의 2개 체인을 동시에 생성하는 병렬 실행 체인을 생성합니다.
combined = RunnableParallel(capital=chain1, area=chain2)

invoke = combined.invoke({"country": "대한민국"})
print(invoke)
print(invoke.get("capital"))
print(invoke.get("area"))