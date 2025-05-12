from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_openai import ChatOpenAI

"""
RunnableWithMessageHistory 기능은 특히 대화형 애플리케이션 또는 복잡한 데이터 처리 작업을 구현할 때 이전 메시지의 맥락을 유지해야 할 필요가 있을 때 중요합니다.
RunnableWithMessageHistory는 애플리케이션의 상태를 유지하고, 사용자 경험을 향상시키며, 더 정교한 응답 메커니즘을 구현할 수 있게 해주는 강력한 도구입니다.
"""
load_dotenv()
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 {ability} 에 능숙한 어시스턴트입니다. 20자 이내로 응답하세요",
        ),
        MessagesPlaceholder(variable_name="history"),   # 대화 기록을 변수로 사용, history가 MessageHistory의 key 가 됨
        ("human", "{input}"),  # 사용자 입력을 변수로 사용
    ]
)
runnable = prompt | model  # 프롬프트와 모델을 연결하여 runnable 객체 생성

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}  # 세션 기록을 저장할 딕셔너리


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    # 주어진 user_id와 conversation_id에 해당하는 세션 기록을 반환합니다.
    if (user_id, conversation_id) not in store:
        # 해당 키가 store에 없으면 새로운 ChatMessageHistory를 생성하여 저장합니다.
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[  # 기존의 "session_id" 설정을 대체하게 됩니다.
        ConfigurableFieldSpec(
            id="user_id",  # get_session_history 함수의 첫 번째 인자로 사용됩니다.
            annotation=str,
            name="User ID",
            description="사용자의 고유 식별자입니다.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",  # get_session_history 함수의 두 번째 인자로 사용됩니다.
            annotation=str,
            name="Conversation ID",
            description="대화의 고유 식별자입니다.",
            default="",
            is_shared=True,
        ),
    ],
)

print(with_message_history.invoke(
    # 능력(ability)과 입력(input)을 포함한 딕셔너리를 전달합니다.
    {"ability": "math", "input": "Hello"},
    # 설정(config) 딕셔너리를 전달합니다.
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
))

print(with_message_history.invoke(
    {"input_messages": "코사인의 의미는 무엇인가요?"},
    # 설정 옵션을 딕셔너리 형태로 전달합니다.
    config={"configurable": {"user_id": "123", "conversation_id": "2"}},
))