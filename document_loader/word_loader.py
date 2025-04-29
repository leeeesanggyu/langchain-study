from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("./data/sample-word-document.docx")  # 문서 로더 초기화

docs = loader.load()  # 문서 로딩

print(len(docs))
