from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="./data/titanic.csv",
    csv_args={
        "delimiter": ",",  # 구분자
        "quotechar": '"',  # 인용 부호 문자
        "fieldnames": [
            "Passenger ID",
            "Survival (1: Survived, 0: Died)",
            "Passenger Class",
            "Name",
            "Sex",
            "Age",
            "Number of Siblings/Spouses Aboard",
            "Number of Parents/Children Aboard",
            "Ticket Number",
            "Fare",
            "Cabin",
            "Port of Embarkation",
        ],  # 필드 이름
    },
)

docs = loader.load()

print(len(docs))
print(docs[0].metadata)

for row in loader.lazy_load():
    print(row)
    break  # 첫 행만 출력
