import pandas as pd

data = pd.read_csv('Data/sem_text_rel_ranked.csv', index_col='Index')

row = data.loc[0]
sent1, sent2 = row['Text'].split("\n")
score = row['Score']

print(data.columns)
print(data.head())
