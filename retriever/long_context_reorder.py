from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

"""
모델의 아키텍처와 상관없이, 10개 이상의 검색된 문서를 포함할 경우 성능이 상당히 저하됩니다.
간단히 말해, 모델이 긴 컨텍스트 중간에 있는 관련 정보에 접근해야 할 때, 제공된 문서를 무시하는 경향이 있습니다.
이때 검색 후 문서의 순서를 재배열하여 성능 저하를 방지할 수 있습니다.
"""

load_dotenv()
logging.langsmith("CH11-Retriever")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts = [
    "이건 그냥 내가 아무렇게나 적어본 글입니다.",
    "사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다.",
    "아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다.",
    "챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다.",
    "챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다.",
    "애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다.",
    "ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다.",
    "비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다.",
    "ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다.",
    "FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다.",
]

retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)

query = "ChatGPT에 대해 무엇을 말해줄 수 있나요?"

docs = retriever.invoke(query)

def format_docs(docs):
    return "\n".join([doc.page_content for i, doc in enumerate(docs)])

# 재정렬
def reorder_documents(docs):
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    combined = format_docs(reordered_docs)
    return combined


template = """Given this text extracts:
{context}

-----

Please answer the following question:
{question}

Answer in the following languages: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question")
        | retriever
        | RunnableLambda(reorder_documents),
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = chain.invoke(
    {"question": "ChatGPT에 대해 무엇을 말해줄 수 있나요?", "language": "KOREAN"}
)
print(answer)