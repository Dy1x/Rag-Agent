import time

from rag.rag_service import RagSummerizeService

rag = RagSummerizeService()

query = "机器人清扫时频繁暂停是什么原因？"

start = time.time()
answer1 = rag.rag_summerize(query)
end = time.time()

print("第一次耗时：", end - start)
print(answer1)
print("=" * 50)

start = time.time()
answer2 = rag.rag_summerize(query)
end = time.time()

print("第二次耗时：", end - start)
print(answer2)
print("=" * 50)

print("两次答案是否一致：", answer1 == answer2)
