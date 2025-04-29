from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

load_dotenv()

print(ENTITY_MEMORY_CONVERSATION_TEMPLATE.template)

llm = ChatOpenAI(temperature=0)

conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm),
)

predict = conversation.predict(input="""
    테디와 셜리는 한 회사에서 일하는 동료입니다.
    테디는 개발자이고 셜리는 디자이너입니다. 
    그들은 최근 회사에서 일하는 것을 그만두고 자신들의 회사를 차릴 계획을 세우고 있습니다.
    """)

# print(predict)

print(conversation.memory.entity_store.store)
