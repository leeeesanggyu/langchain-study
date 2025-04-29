from langchain_core.documents import Document

document = Document("안녕하세요? 이건 랭체인의 도큐먼트 입니다")

print(document.__dict__)

document.metadata["source"] = "TeddyNote"
document.metadata["page"] = 1
document.metadata["author"] = "SangGyu"

print(document.__dict__)
