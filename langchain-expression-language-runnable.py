from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
logging.langsmith("langchain-expression-language-runnable")

prompt = PromptTemplate.from_template("{num} 의 10배는?")

runnable_chain = {"num": RunnablePassthrough()} | prompt | ChatOpenAI()

print(runnable_chain.invoke(10))
