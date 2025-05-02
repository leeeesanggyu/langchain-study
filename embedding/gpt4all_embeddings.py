from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings

load_dotenv()
gpt4all_embd = GPT4AllEmbeddings()

text = (
    "임베딩 테스트를 하기 위한 샘플 문장입니다."
)

doc_result = gpt4all_embd.embed_documents([text])
print(doc_result)

