from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain_teddynote import logging
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
logging.langsmith("CH04-Models")

template = """
<|system|>
You are a helpful assistant.<|end|>
<|user|>
{question}<|end|>
<|assistant|>
"""
prompt = PromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    max_new_tokens=256,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    temperature=0.1,
)

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"question": "what is the capital of South Korea?"})
print(response)
