from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.langsmith("CH04-Models")

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt = PromptTemplate.from_template("{country} 에 대해서 200자 내외로 요약해줘")

chain = prompt | llm

set_llm_cache(InMemoryCache())

response = chain.invoke({"country": "한국"})
print(response.content)

response = chain.invoke({"country": "한국"})
print(response.content)
