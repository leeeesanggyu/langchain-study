from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

"""
각 종류 별 PDF Loader: https://wikidocs.net/253707
"""
FILE_PATH = "./example.pdf"

loader = PyPDFLoader(FILE_PATH)
# docs = loader.load()
# print(docs)
# print(len(docs))
# print(docs[0].page_content)

text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=3)
docs = loader.load_and_split(text_splitter=text_splitter)

print(len(docs))
print(docs[0].page_content)
