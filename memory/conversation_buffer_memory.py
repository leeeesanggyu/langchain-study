from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("Memory")

memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    inputs={
        "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
    },
    outputs={
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
    },
)
memory.save_context(
    inputs={
        "human": "네, 신분증을 준비했습니다. 이제 무엇을 해야 하나요?"
    },
    outputs={
        "ai": "감사합니다. 신분증 앞뒤를 명확하게 촬영하여 업로드해 주세요. 이후 본인 인증 절차를 진행하겠습니다."
    },
)
memory.save_context(
    inputs={
        "human": "사진을 업로드했습니다. 본인 인증은 어떻게 진행되나요?"
    },
    outputs={
        "ai": "업로드해 주신 사진을 확인했습니다. 이제 휴대폰을 통한 본인 인증을 진행해 주세요. 문자로 발송된 인증번호를 입력해 주시면 됩니다."
    },
)
memory.save_context(
    inputs={
        "human": "인증번호를 입력했습니다. 계좌 개설은 이제 어떻게 하나요?"
    },
    outputs={
        "ai": "본인 인증이 완료되었습니다. 이제 원하시는 계좌 종류를 선택하고 필요한 정보를 입력해 주세요. 예금 종류, 통화 종류 등을 선택할 수 있습니다."
    },
)

# print(memory.load_memory_variables({})["history"])

llm = ChatOpenAI(temperature=0)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
)

response = conversation.predict(
    input="이전 답변을 불렛포인트 형식으로 정리하여 알려주세요."
)
print(response)
