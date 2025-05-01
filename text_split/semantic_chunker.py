from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

with open("./example.txt") as f:
    file = f.read()

text_splitter = SemanticChunker(OpenAIEmbeddings())
# chunks = text_splitter.split_text(file)

# print(len(chunks))
# print(chunks[1])

docs = text_splitter.create_documents([file])
print(len(docs))
print(docs[0].page_content)  # 분할된 문서 중 첫 번째 문서의 내용을 출력합니다.

