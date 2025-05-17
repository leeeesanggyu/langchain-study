import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser

dotenv.load_dotenv()

########## 1. 도구를 정의합니다 ##########

### 1-1. Search 도구 ###
# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=6은 검색 결과를 6개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=6)

### 1-2. PDF 문서 검색 도구 (Retriever) ###
# PDF 파일 로드. 파일의 경로 입력
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성합니다.
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",  # 도구의 이름을 입력합니다.
    description="use this tool to search information from the PDF document",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)

### 1-3. tools 리스트에 도구 목록을 추가합니다 ###
# tools 리스트에 search와 retriever_tool을 추가합니다.
tools = [search, retriever_tool]

########## 2. LLM 을 정의합니다 ##########
# LLM 모델을 생성합니다.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

########## 3. Prompt 를 정의합니다 ##########

# Prompt 를 정의합니다 - 이 부분을 수정할 수 있습니다!
# Prompt 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
            "If you can't find the information from the PDF document, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

########## 4. Agent 를 정의합니다 ##########

# 에이전트를 생성합니다.
# llm, tools, prompt를 인자로 사용합니다.
agent = create_tool_calling_agent(llm, tools, prompt)

########## 5. AgentExecutor 를 정의합니다 ##########

# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

########## 6. 채팅 기록을 수행하는 메모리를 추가합니다. ##########

# session_id 를 저장할 딕셔너리 생성
store = {}


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

########## 7. Agent 파서를 정의합니다. ##########
agent_stream_parser = AgentStreamParser()

########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########
#
# # 질의에 대한 답변을 출력합니다.
# response = agent_with_chat_history.stream(
#     {"input": "구글이 앤스로픽에 투자한 금액을 문서에서 찾아줘"},
#     # 세션 ID를 설정합니다.
#     # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
#     config={"configurable": {"session_id": "abc123"}},
# )
#
# for step in response:
#     agent_stream_parser.process_agent_steps(step)
#
# ########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########
#
# # 질의에 대한 답변을 출력합니다.
# response = agent_with_chat_history.stream(
#     {"input": "이전의 답변을 영어로 번역해 주세요"},
#     # 세션 ID를 설정합니다.
#     # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
#     config={"configurable": {"session_id": "abc123"}},
# )
#
# for step in response:
#     agent_stream_parser.process_agent_steps(step)

########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {
        "input": "2024년 프로야구 플레이오프 진출 5개팀을 검색해서 알려주세요. 한글로 답변하세요"
    },
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc456"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)

########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "이전의 답변에 한국 시리즈 일정을 추가하세요."},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc456"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
