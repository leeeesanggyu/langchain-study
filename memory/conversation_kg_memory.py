from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationKGMemory

load_dotenv()

llm = ChatOpenAI(temperature=0)

template = """
The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. 
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:
"""
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

conversation_with_kg = ConversationChain(
    llm=llm, prompt=prompt, memory=ConversationKGMemory(llm=llm)
)

conversation_with_kg.predict(
    input="My name is Teddy. Shirley is a coworker of mine, and she's a new designer at our company."
)

print(conversation_with_kg.memory.load_memory_variables({"input": "who is Shirley?"}))
