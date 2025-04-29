from langchain_community.document_loaders import JSONLoader

"""
JSON 데이터의 메시지 키 내 content 필드 아래의 값을 추출하고 싶다고 가정하였을 때, 
아래와 같이 JSONLoader를 통해 쉽게 수행할 수 있습니다.
"""
# JSONLoader 생성
loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".[].phoneNumbers",
    text_content=False,
)

# 문서 로드
docs = loader.load()

# 결과 출력
print(docs)
