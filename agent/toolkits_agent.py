from dotenv import load_dotenv
from langchain_teddynote import logging
import os
from langchain_community.agent_toolkits import FileManagementToolkit
from typing import List, Dict
from langchain_teddynote.tools import GoogleNews
from langchain.agents import tool
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser

"""LangChain 프레임워크를 사용하는 가장 큰 이점은 3rd-party integration 되어 있는 다양한 기능들입니다.
그 중 Toolkits 는 다양한 도구를 통합하여 제공합니다."""

load_dotenv()
logging.langsmith("CH15-Agent-Toolkits")

# 파일 관리 도구 생성
tools = FileManagementToolkit(root_dir="tmp").get_tools()

working_directory = "tmp"

# 최신 뉴스 검색 도구를 정의합니다.
@tool
def latest_news(k: int = 5) -> List[Dict[str, str]]:
    """Look up latest news"""
    # GoogleNews 객체를 생성합니다.
    news_tool = GoogleNews()
    # 최신 뉴스를 검색하고 결과를 반환합니다. k는 반환할 뉴스 항목의 수입니다.
    return news_tool.search_latest(k=k)


# FileManagementToolkit을 사용하여 파일 관리 도구들을 가져옵니다.
tools = FileManagementToolkit(
    root_dir=str(working_directory),
).get_tools()

# 최신 뉴스 검색 도구를 tools 리스트에 추가합니다.
tools.append(latest_news)

# 모든 도구들이 포함된 tools 리스트를 출력합니다.
print(f"[tools 리스트를 출력] {tools}")

# session_id 를 저장할 딕셔너리 생성
store = {}

# 프롬프트 생성
# 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `latest_news` tool to find latest news. "
            "Make sure to use the `file_management` tool to manage files. ",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# LLM 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# Agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)


# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대화 session_id
    get_session_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)

agent_stream_parser = AgentStreamParser()

result = agent_with_chat_history.stream(
    {
        "input": "최신 뉴스 5개를 검색하고, 각 뉴스의 제목을 파일명으로 가지는 파일을 생성하고(.txt), "
        "파일의 내용은 뉴스의 내용과 url을 추가하세요. "
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)

result = agent_with_chat_history.stream(
    {
        "input": "이전에 생성한 파일 제목 맨 앞에 제목에 어울리는 emoji를 추가하여 파일명을 변경하세요. "
        "파일명도 깔끔하게 변경하세요. "
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)

result = agent_with_chat_history.stream(
    {
        "input": "이전에 생성한 모든 파일을 `news` 폴더를 생성한 뒤 해당 폴더에 모든 파일을 복사하세요. "
        "내용도 동일하게 복사하세요. "
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)

result = agent_with_chat_history.stream(
    {"input": "news 폴더를 제외한 모든 .txt 파일을 삭제하세요."},
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
