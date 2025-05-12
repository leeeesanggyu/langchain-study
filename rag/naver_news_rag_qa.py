from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_teddynote.messages import stream_response
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
docs = loader.load()
print(f"문서의 수: {len(docs)}")
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"잘린 문서의 수: {len(splits)}")

vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 
만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:
"""
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.stream("부영그룹의 출산 장려 정책에 대해 설명해주세요.")
print("\n== 부영그룹의 출산 장려 정책에 대해 설명해주세요.")
stream_response(answer)

answer2 = rag_chain.stream("부영그룹은 출산 직원에게 얼마의 지원을 제공하나요?")
print("\n\n== 부영그룹은 출산 직원에게 얼마의 지원을 제공하나요?")
stream_response(answer2)

answer3 = rag_chain.stream("정부의 저출생 대책을 bullet points 형식으로 작성해 주세요.")
print("\n\n== 정부의 저출생 대책을 bullet points 형식으로 작성해 주세요.")
stream_response(answer3)

answer4 = rag_chain.stream("부영그룹의 임직원 숫자는 몇명인가요?")
print("\n\n== 부영그룹의 임직원 숫자는 몇명인가요?")
stream_response(answer4)

