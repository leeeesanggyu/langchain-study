import os
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

model_id = "beomi/llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)  # 지정된 모델의 토크나이저를 로드합니다.
model = AutoModelForCausalLM.from_pretrained(model_id)  # 지정된 모델을 로드합니다.

# 텍스트 생성 파이프라인을 생성하고, 최대 생성할 새로운 토큰 수를 10으로 설정합니다.
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
)

hf = HuggingFacePipeline(pipeline=pipe) # HuggingFacePipeline 객체를 생성하고, 생성된 파이프라인을 전달합니다.

template = """
Answer the following question in Korean.
#Question: 
{question}
#Answer: 
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf | StrOutputParser()

print(
    chain.invoke({"question":  "대한민국의 수도는 어디야?"})
)