from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("./example.txt") as f:
    file = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # 청크 크기를 매우 작게 설정합니다. 예시를 위한 설정입니다.
    chunk_size=250,
    # 청크 간의 중복되는 문자 수를 설정합니다.
    chunk_overlap=50,
    # 문자열 길이를 계산하는 함수를 지정합니다.
    length_function=len,
    # 구분자로 정규식을 사용할지 여부를 설정합니다.
    is_separator_regex=False,
)

texts = text_splitter.create_documents([file])
print(texts[0].page_content)
print("===" * 20)
print(texts[1].page_content)
print("===" * 20)
print(texts[2].page_content)
