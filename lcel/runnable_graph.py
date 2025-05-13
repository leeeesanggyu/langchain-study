from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

vectorstore = FAISS.from_texts(
    ["SangGyu is an AI engineer who loves programming!"], # 텍스트 데이터로부터 FAISS 벡터 저장소를 생성합니다.
    embedding=OpenAIEmbeddings(),
)

# 벡터 저장소를 기반으로 retriever를 생성합니다.
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}  

Question: {question}"""

# 템플릿을 기반으로 ChatPromptTemplate을 생성합니다.
prompt = ChatPromptTemplate.from_template(
    template
)

model = ChatOpenAI(model="gpt-4o-mini")  # ChatOpenAI 모델을 초기화합니다.

# chain 을 생성합니다.
chain = (
    # 검색 컨텍스트와 질문을 지정합니다.
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt  # 프롬프트를 생성합니다.
    | model  # 언어 모델을 실행합니다.
    | StrOutputParser()  # 출력 결과를 문자열로 파싱합니다.
)

print(chain.get_graph().nodes)

print("=" * 30)
print(chain.get_graph().edges)

print("=" * 30)
print(chain.get_graph().print_ascii())

print("=" * 30)
print(chain.get_prompts())