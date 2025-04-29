import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)

docs = loader.load()
print(f"문서의 수: {len(docs)}")
print(docs)

# ssl 인증 우회
loader.requests_kwargs = {"verify": False}
docs = loader.load()
print(docs)

"""
여러 웹페이지를 한 번에 로드할 수도 있습니다. 
이를 위해 urls의 리스트를 로더에 전달하면, 전달된 urls의 순서대로 문서 리스트를 반환합니다.
"""
loader = WebBaseLoader(
    web_paths=[
        "https://n.news.naver.com/article/437/0000378416",
        "https://n.news.naver.com/mnews/hotissue/article/092/0002340014?type=series&cid=2000063",
    ],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content[:500])
print("===" * 10)
print(docs[1].page_content[:500])

"""
IP 차단을 우회하기 위해 때때로 프록시를 사용할 필요가 있을 수 있습니다.
프록시를 사용하려면 로더(및 그 아래의 requests)에 프록시 딕셔너리를 전달할 수 있습니다.
"""
loader = WebBaseLoader(
    "https://www.google.com/search?q=parrots",
    proxies={
        "http": "http://{username}:{password}:@proxy.service.com:6666/",
        "https": "https://{username}:{password}:@proxy.service.com:6666/",
    },
    # 웹 기반 로더 초기화
    # 프록시 설정
)

docs = loader.load()