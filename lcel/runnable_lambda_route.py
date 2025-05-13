from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

"""
RunnableBranch 는 입력에 따라 동적으로 로직을 라우팅할 수 있는 강력한 도구입니다. 이를 통해 개발자는 입력 데이터의 특성에 기반하여 다양한 처리 경로를 유연하게 정의 할 수 있습니다.
RunnableBranch 는 복잡한 의사 결정 트리를 간단하고 직관적인 방식으로 구현할 수 있도록 도와줍니다. 이는 코드의 가독성과 유지보수성을 크게 향상시키며, 로직의 모듈화와 재사용성을 촉진합니다.
또한, RunnableBranch 는 런타임에 동적으로 분기 조건을 평가하고 적절한 처리 루틴을 선택할 수 있어, 시스템의 적응력과 확장성을 높여줍니다.
이러한 특징들로 인해 RunnableBranch는 다양한 도메인에서 활용될 수 있으며, 특히 입력 데이터의 다양성과 변동성이 큰 애플리케이션 개발에 매우 유용합니다. RunnableBranch 를 효과적으로 활용하면 코드의 복잡성을 줄이고, 시스템의 유연성과 성능을 향상시킬 수 있습니다.
"""

load_dotenv()

prompt = PromptTemplate.from_template(
    """주어진 사용자 질문을 `수학`, `과학`, 또는 `기타` 중 하나로 분류하세요. 한 단어 이상으로 응답하지 마세요.

<question>
{question}
</question>

Classification:"""
)

# 체인을 생성합니다.
llm = ChatOpenAI(model="gpt-4o-mini")
chain = (
        prompt
        | llm
        | StrOutputParser()  # 문자열 출력 파서를 사용합니다.
)

math_chain = (
        PromptTemplate.from_template(
            """You are an expert in math. \
    Always answer questions starting with "깨봉선생님께서 말씀하시기를..". \
    Respond to the following question:
    
    Question: {question}
    Answer:"""
        )
        | llm
)

science_chain = (
        PromptTemplate.from_template(
            """You are an expert in science. \
    Always answer questions starting with "아이작 뉴턴 선생님께서 말씀하시기를..". \
    Respond to the following question:
    
    Question: {question}
    Answer:"""
        )
        | llm
)

general_chain = (
        PromptTemplate.from_template(
            """Respond to the following question concisely:
    
    Question: {question}
    Answer:"""
        )
        | llm
)

def route(info):
    if "수학" in info["topic"].lower():
        return math_chain
    elif "과학" in info["topic"].lower():
        return science_chain
    else:
        return general_chain


full_chain = (
    {"topic": chain, "question": itemgetter("question")}
    | RunnableLambda(
        # 경로를 지정하는 함수를 인자로 전달합니다.
        route
    )
    | StrOutputParser()
)

print(full_chain.invoke({"question": "2+2 는 무엇인가요?"}))
print(full_chain.invoke({"question": "작용 반작용의 법칙은 무엇인가요?"}))
print(full_chain.invoke({"question": "Google은 어떤 회사인가요?"}))

