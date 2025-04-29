from langchain_community.document_loaders import UnstructuredPowerPointLoader

loader = UnstructuredPowerPointLoader("./data/sample-ppt.pptx")

docs = loader.load()

print(len(docs))
