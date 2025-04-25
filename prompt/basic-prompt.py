from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logging.langsmith("basic-prompt")

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

chain = (
    chat_template
    | ChatOpenAI()
    | StrOutputParser()
)

print(chain.invoke({"name": "상규", "user_input": "당신의 이름은 무엇입니까?"}))


