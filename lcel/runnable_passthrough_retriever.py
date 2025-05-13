from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# 텍스트로부터 FAISS 벡터 저장소를 생성합니다.
vectorstore = FAISS.from_texts(
    [
        "상규는 랭체인 주식회사에서 근무를 하였습니다.",
        "동렬은 상규와 같은 회사에서 근무하였습니다.",
        "상규의 직업은 개발자입니다.",
        "동렬의 직업은 디자이너입니다.",
    ],
    embedding=OpenAIEmbeddings(),
)

# 벡터 저장소를 검색기로 사용합니다.
retriever = vectorstore.as_retriever()

# 템플릿을 정의합니다.
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
# 템플릿으로부터 채팅 프롬프트를 생성합니다.
prompt = ChatPromptTemplate.from_template(template)

# ChatOpenAI 모델을 초기화합니다.
model = ChatOpenAI(model_name="gpt-4o-mini")


# 문서를 포맷팅하는 함수
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


# 검색 체인을 구성합니다.
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 검색 체인을 실행하여 질문에 대한 답변을 얻습니다.
print(retrieval_chain.invoke("상규의 직업은 무엇입니까?"))
print(retrieval_chain.invoke("동렬의 직업은 무엇입니까?"))
