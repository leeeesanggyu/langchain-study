from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

sentence1 = "안녕하세요? 반갑습니다."
sentence2 = "안녕하세요? 반갑습니다!"
sentence3 = "안녕하세요? 만나서 반가워요."
sentence4 = "Hi, nice to meet you."
sentence5 = "I like to eat apples."

embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

sentences = [sentence1, sentence2, sentence3, sentence4, sentence5]
embedded_sentences = embeddings_1024.embed_documents(sentences)

def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]

for i, sentence in enumerate(embedded_sentences):
    for j, other_sentence in enumerate(embedded_sentences):
        if i < j:
            print(
                f"[유사도 {similarity(sentence, other_sentence):.4f}] {sentences[i]} \t <=====> \t {sentences[j]}"
            )
