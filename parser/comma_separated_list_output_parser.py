from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
logging.langsmith("CH03-OutputParser")

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 입력 변수로 'subject' 사용
    partial_variables={"format_instructions": format_instructions}, # 부분 변수로 형식 지침 사용
)

model = ChatOpenAI(temperature=0)

chain = prompt | model | output_parser

for s in chain.stream({"subject": "대한민국 관광명소"}):
    print(s)  # 스트림의 내용을 출력합니다.