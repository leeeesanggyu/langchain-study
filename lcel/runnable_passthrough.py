from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

runnable = RunnableParallel(
    passed=RunnablePassthrough(),   # 전달된 입력을 그대로 반환하는 Runnable을 설정합니다.
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),  # 입력의 "num" 값에 3을 곱한 결과를 반환하는 Runnable을 설정합니다.
    modified=lambda x: x["num"] + 1,    # 입력의 "num" 값에 1을 더한 결과를 반환하는 Runnable을 설정합니다.
)

print(runnable.invoke({"num": 1}))
