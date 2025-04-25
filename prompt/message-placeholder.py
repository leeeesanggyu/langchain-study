from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
logging.langsmith("basic-prompt")

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다."),
    ]
)

chain = (
        chat_prompt
        | ChatOpenAI()
        | StrOutputParser()
)

invoke = chain.invoke({
    "word_count": 5,
    "conversation": [
        ("human", "안녕하세요! 저는 오늘 새로 입사한 상규 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요")
    ]
})
print(invoke)
