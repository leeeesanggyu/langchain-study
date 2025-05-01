from langchain_text_splitters import CharacterTextSplitter

with open("example.txt") as f:
    file = f.read()

text_splitter = CharacterTextSplitter(
    # separator=" ",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

metadatas = [
    {"document": 1},
    {"document": 2},
]  # 문서에 대한 메타데이터 리스트를 정의합니다.

texts = text_splitter.create_documents(
    [file, file],
    metadatas=metadatas
)
print(texts[1])
