from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("CH02-Prompt")

prompt = ChatPromptTemplate.from_template(
    "주어진 내용을 바탕으로 다음 문장을 요약하세요. 답변은 반드시 한글로 작성하세요\n\n"
    "CONTEXT: {context}\n\n"
    "SUMMARY:"
)

hub.push("leeeesanggyu/simple-summary-korean", prompt)

pulled_prompt = hub.pull("leeeesanggyu/simple-summary-korean")
print(pulled_prompt)
