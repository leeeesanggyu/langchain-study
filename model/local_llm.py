from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.gpt4all import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler

prompt = ChatPromptTemplate.from_template("""
<s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>
<s>Human: {question}</s>
<s>Assistant:
"""
)

local_path = "/Users/salgu/Library/Application Support/nomic.ai/GPT4All/Llama-3.2-1B-Instruct-Q4_0.gguf"
llm = GPT4All(
    model=local_path,
    backend="gpu",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"question": "대한민국의 수도는 어디인가요?"})
