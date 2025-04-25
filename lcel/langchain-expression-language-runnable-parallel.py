from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
logging.langsmith("langchain-expression-language-runnable-parallel")

chain1 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 수도는?")
    | ChatOpenAI()
    | StrOutputParser()
)
chain2 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 면적은?")
    | ChatOpenAI()
    | StrOutputParser()
)

combined_chain = RunnableParallel(capital=chain1, area=chain2)
print(combined_chain.invoke("대한민국"))

