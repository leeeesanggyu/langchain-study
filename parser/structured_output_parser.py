from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
logging.langsmith("CH03-OutputParser")

# 사용자의 질문에 대한 답변
response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"),
    ResponseSchema(
        name="source",
        description="사용자의 질문에 답하기 위해 사용된 `출처`, `웹사이트주소` 이여야 합니다.",
    ),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)
chain = prompt | model | output_parser

for s in chain.stream({"question": "세종대왕의 업적은 무엇인가요?"}):
    print(s)

