from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("LCEL-Advanced")

# 프롬프트 템플릿을 정의합니다.
prompt1 = ChatPromptTemplate.from_template("{topic} 에 대해 짧게 한글로 설명해주세요.")
prompt2 = ChatPromptTemplate.from_template(
    "{sentence} 를 emoji를 활용한 인스타그램 게시글로 만들어주세요."
)


@chain
def custom_chain(text):
    # 첫 번째 프롬프트, ChatOpenAI, 문자열 출력 파서를 연결하여 체인을 생성합니다.
    chain1 = prompt1 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    output1 = chain1.invoke({"topic": text})

    # 두 번째 프롬프트, ChatOpenAI, 문자열 출력 파서를 연결하여 체인을 생성합니다.
    chain2 = prompt2 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    # 두 번째 체인을 호출하여 파싱된 첫 번째 결과를 전달하고 최종 결과를 반환합니다.
    return chain2.invoke({"sentence": output1})


print(custom_chain.invoke("양자역학"))