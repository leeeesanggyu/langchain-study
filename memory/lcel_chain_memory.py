from dotenv import load_dotenv
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

"""
임의의 체인에 메모리를 추가하는 방법을 보여줍니다. 현재 메모리 클래스를 사용할 수 있지만 수동으로 연결해야 합니다
"""

load_dotenv()

model = ChatOpenAI()

# 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

runnable = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
)

# 대화형 프롬프트를 생성합니다. 이 프롬프트는 시스템 메시지, 이전 대화 내역, 그리고 사용자 입력을 포함합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = runnable | prompt | model

# chain 객체의 invoke 메서드를 사용하여 입력에 대한 응답을 생성합니다.
input = "만나서 반갑습니다. 제 이름은 상규입니다."
response = chain.invoke({"input": input})
print(response.content)  # 생성된 응답을 출력합니다.

# 입력된 데이터와 응답 내용을 메모리에 저장합니다.
memory.save_context(
    {"human": input},
    {"ai": response.content}
)

# 저장된 대화기록을 출력합니다.
print(memory.load_memory_variables({}))

# 이름을 기억하고 있는지 추가 질의합니다.
response = chain.invoke({"input": "제 이름이 무엇이었는지 기억하세요?"})
print(response.content)
