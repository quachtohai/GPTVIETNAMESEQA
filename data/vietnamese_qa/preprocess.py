import pandas as pd


df = pd.read_json("./data/vietnamese_qa/QA.json")

df = df.drop(columns=['en_question', 'en_answer'])

df["summary"] = df["vi_question"] +"BEGIN"+ df["vi_answer"] + "END"

df = df.drop (columns=["vi_question", "vi_answer"])
df.to_csv("QA.csv", sep='\t', encoding='utf-8')
print(df.head())