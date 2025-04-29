from langchain_community.document_loaders import UnstructuredExcelLoader

loader = UnstructuredExcelLoader("./data/titanic.xlsx", mode="elements")

docs = loader.load()

print(len(docs))
print(docs[0].page_content[:200])
