import warnings

from dotenv import load_dotenv
from langchain_teddynote.tools import GoogleNews
from langchain.tools import tool
from typing import List, Dict

warnings.filterwarnings("ignore")
load_dotenv()

news_tool = GoogleNews()

# 최신 뉴스 검색
print(news_tool.search_latest(k=5))

# 키워드로 뉴스 검색
keyword = news_tool.search_by_keyword("윤석열", k=5)
print(keyword)


# 키워드로 뉴스 검색하는 도구 생성
@tool
def search_keyword(query: str) -> List[Dict[str, str]]:
    """Look up news by keyword"""
    print(query)
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


# 실행 결과
search_keyword.invoke({"query": "AI 투자"})
