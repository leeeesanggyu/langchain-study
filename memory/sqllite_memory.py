from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.utils import ConfigurableFieldSpec


load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),  # 질문
    ]
)

chain = prompt | ChatOpenAI(model_name="gpt-4o") | StrOutputParser()

def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///sqlite.db",
    )

config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        default="",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="Conversation ID",
        description="Unique identifier for a conversation.",
        default="",
        is_shared=True,
    ),
]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,  # 대화 기록을 가져오는 함수를 설정합니다.
    input_messages_key="question",  # 입력 메시지의 키를 "question"으로 설정
    history_messages_key="chat_history",  # 대화 기록 메시지의 키를 "history"로 설정
    history_factory_config=config_fields,  # 대화 기록 조회시 참고할 파라미터를 설정합니다.
)

config = {
    "configurable": {
        "user_id": "user1",
        "conversation_id": "conversation1"
    }
}

print(chain_with_history.invoke({"question": "안녕 반가워, 내 이름은 상규야"},config))
print(chain_with_history.invoke({"question": "내 이름이 뭐라고?"}, config))
