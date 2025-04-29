from langchain_teddynote.document_loaders import HWPLoader

loader = HWPLoader("./data/디지털 정부혁신 추진계획.hwp")

docs = loader.load()
print(docs[0].page_content[:1000])
