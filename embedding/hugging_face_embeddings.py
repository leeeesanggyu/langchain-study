import os
import warnings

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

warnings.filterwarnings("ignore")

load_dotenv()

texts = [
    "안녕, 만나서 반가워.",
    "LangChain simplifies the process of building applications with large language models",
    "랭체인 한국어 튜토리얼은 LangChain의 공식 문서, cookbook 및 다양한 실용 예제를 바탕으로 하여 사용자가 LangChain을 더 쉽고 효과적으로 활용할 수 있도록 구성되어 있습니다. ",
    "LangChain은 초거대 언어모델로 애플리케이션을 구축하는 과정을 단순화합니다.",
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.",
]

model_name = "intfloat/multilingual-e5-large-instruct"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

embedded_documents = hf_embeddings.embed_documents(texts)

print("[HuggingFace Endpoint Embedding]")
print(f"Model: \t\t{model_name}")
print(f"Dimension: \t{len(embedded_documents[0])}")

embedded_query = hf_embeddings.embed_query("LangChain 에 대해서 알려주세요.")
print(f"Query Embedded: \t\t{embedded_query}")


print("[Query] LangChain 에 대해서 알려주세요.\n====================================")
sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] {texts[idx]}")
    print()
