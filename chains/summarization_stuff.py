from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_teddynote.callbacks import StreamingCallback

"""
Stuff: 단순히 모든 문서를 단일 프롬프트로 "넣는" 방식입니다. 이는 가장 간단한 접근 방식입니다.
"""

load_dotenv()

loader = PyPDFLoader("./data/example_resume.pdf")
docs = loader.load()
print(f"총 글자수: {len(docs[0].page_content)}")

prompt = hub.pull("teddynote/summary-stuff-documents-korean")
prompt.pretty_print()

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    temperature=0,
    callbacks=[StreamingCallback()],
)

stuff_chain = create_stuff_documents_chain(llm, prompt)
answer = stuff_chain.invoke({"context": docs})
print(f"answer = {answer}")