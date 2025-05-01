import warnings

from langchain_text_splitters import CharacterTextSplitter

with open("./example.txt") as f:
    file = f.read()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=0,
)

texts = text_splitter.split_text(file)
print(len(texts))
print(texts[0])
