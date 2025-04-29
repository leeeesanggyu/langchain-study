from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
logging.langsmith("CH05-Memory")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Question:\n{question}"),
    ]
)

llm = ChatOpenAI(model_name="gpt-4o")

chain = prompt | llm | StrOutputParser()

# 세션 기록을 저장할 딕셔너리
store = {}

def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",
    history_messages_key="chat_history",
)

print(chain_with_history.invoke(
    {"question": "나의 이름은 상규입니다."},
    config={"configurable": {"session_id": "aaaa"}},
))
print(chain_with_history.invoke(
    {"question": "내 이름이 뭐라고?"},
    config={"configurable": {"session_id": "aaaa"}},
))
print(chain_with_history.invoke(
    {"question": "내 이름이 뭐라고?"},
    config={"configurable": {"session_id": "bbbb"}},
))

