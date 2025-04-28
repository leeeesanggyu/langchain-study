from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()
logging.langsmith("CH03-OutputParser")

model = ChatOpenAI(temperature=0, model_name="gpt-4o")

class Topic(BaseModel):
    description: str = Field(description="주제에 대한 간결한 설명")
    hashtags: str = Field(description="해시태그 형식의 키워드(2개 이상)")

parser = JsonOutputParser(pydantic_object=Topic)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 어시스턴트 입니다. 질문에 간결하게 답변하세요."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

print(chain.invoke({"question": "지구 온난화의 심각성 대해 알려주세요."}))
