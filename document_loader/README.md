# 문서 로더(Document Loader)

LangChain 의 문서 로더(Document Loader)를 사용해 다양한 형식의 데이터 파일로부터 문서로 로드 할 수 있습니다.

로드한 문서는 `Document` 객체로 로드되며 `page_content` 에는 내용이 `metadata` 에는 메타데이터를 포함합니다.

예를 들어 간단한 `pdf` 파일을 로드하거나, Word, CSV, Excel, JSON 등을 로드하기 위한 문서 로더가 각각 존재합니다.

문서 로더는 `load()`, `aload()`, `lazy_load()` 등 다양한 기능을 제공합니다.

## 주요 Loader

- PyPDFLoader: PDF 파일을 로드하는 로더입니다.
- CSVLoader: CSV 파일을 로드하는 로더입니다.
- UnstructuredHTMLLoader: HTML 파일을 로드하는 로더입니다.
- JSONLoader: JSON 파일을 로드하는 로더입니다.
- TextLoader: 텍스트 파일을 로드하는 로더입니다.
- DirectoryLoader: 디렉토리를 로드하는 로더입니다.
