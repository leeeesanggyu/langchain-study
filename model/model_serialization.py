from dotenv import load_dotenv
from langchain_core.load import dumpd
from langchain_teddynote import logging
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
logging.langsmith("CH04-Models")

prompt = PromptTemplate.from_template("{fruit}의 색상이 무엇입니까?")
print(f"ChatOpenAI: {ChatOpenAI.is_lc_serializable()}")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print(f"ChatOpenAI: {llm.is_lc_serializable()}")

chain = prompt | llm
print(f"ChatOpenAI: {chain.is_lc_serializable()}")

dumpd_chain = dumpd(chain)
print(dumpd_chain)
print(type(dumpd_chain))