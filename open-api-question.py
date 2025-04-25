from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
from langchain_teddynote.models import MultiModal

load_dotenv()

logging.langsmith("langsmith-study")

llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-3.5-turbo",
)

"""
질의
"""
question = "대한민국의 수도는 어디인가요?"
print(f"[답변]: {llm.invoke(question)}")

"""
Stream 질의
"""
answer = llm.stream("대한민국의 아름다운 관광지 10곳과 주소를 알려주세요!")
for token in answer:
    print(token.content, end="", flush=True)
stream_response(answer)

"""
이미지 질의
"""
llm = ChatOpenAI(
    temperature=0.1,
    max_tokens=2048,
    model_name="gpt-4o",
)

# 멀티모달 객체 생성
multimodal_llm = MultiModal(llm)

# 샘플 이미지 주소(웹사이트로 부터 바로 인식)
IMAGE_URL = "https://t3.ftcdn.net/jpg/03/77/33/96/360_F_377339633_Rtv9I77sSmSNcev8bEcnVxTHrXB4nRJ5.jpg"

# 이미지 파일로 부터 질의
answer = multimodal_llm.stream(IMAGE_URL)
# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
stream_response(answer)

"""
프롬프트 질의
"""
system_prompt = """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 
당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""

user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

# 멀티모달 객체 생성
multimodal_llm_with_prompt = MultiModal(
    llm, system_prompt=system_prompt, user_prompt=user_prompt
)

# 로컬 PC 에 저장되어 있는 이미지의 경로 입력
IMAGE_PATH_FROM_FILE = "https://storage.googleapis.com/static.fastcampus.co.kr/prod/uploads/202212/080345-661/kwon-01.png"

# 이미지 파일로 부터 질의(스트림 방식)
answer = multimodal_llm_with_prompt.stream(IMAGE_PATH_FROM_FILE)

# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
stream_response(answer)