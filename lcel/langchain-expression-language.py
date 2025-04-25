from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logging.langsmith("langchain-expression-language")

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 [FORMAT]에 영어 회화를 작성해 주세요.

상황:
{question}

FORMAT:
- 영어 회화:
- 한글 해석:
"""
prompt = PromptTemplate.from_template(template)
model = ChatOpenAI(model_name="gpt-4-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

answer = chain.stream({"question": "저는 식당에 가서 음식을 주문하고 싶어요"})
stream_response(answer)