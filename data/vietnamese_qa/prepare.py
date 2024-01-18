import os
import requests
import numpy as np
import pandas as pd
import tiktoken

df = pd.read_json('./data/vietnamese_qa/QA.json', encoding='utf-8')
df = df.drop(columns=['en_question', 'en_answer'])

df["summary"] = df["vi_question"] +"BEGIN"+ df["vi_answer"] + "END"

df = df.drop (columns=["vi_question", "vi_answer"])
data = df['summary'].str.cat(sep='\n')
n = len(data)
# print (data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))