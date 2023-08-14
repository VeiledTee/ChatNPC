import pandas as pd
import numpy as np

from test import get_sentence_embedding, count_negations

TO_CLEAN: list = ["Data/contradiction-dataset.csv"]

for file in TO_CLEAN:
    df = pd.read_csv(file)
    df['label'] = df['gold_label'].str.lower()
    df["label"] = np.where(df["label"] == "contradiction", 1, 0)
    print("labels complete")
    # Apply the get_sentence_embedding function to generate embeddings
    df["sentence1_embeddings"] = df["sentence1"].apply(lambda x: get_sentence_embedding(x))
    print("sentence1_embeddings complete")
    df["sentence2_embeddings"] = df["sentence2"].apply(lambda x: get_sentence_embedding(x))
    print("sentence2_embeddings complete")
    # Apply the count_negations function to generate negation counts
    df["negation"] = df[["sentence1", "sentence2"]].apply(lambda x: count_negations(x.tolist()), axis=1)
    print("negation complete")
    # for index, row in df.iterrows():
    #     print(f"Index: {index}")
    #     print(f"Sentence 1: {row['sentence1']}")
    #     print(f"Sentence 2: {row['sentence2']}")
    #     print(f"Label: {row['label']}")
    #     print(f"Negation Count: {row['negation']}")
    #     print(f"Sentence 1 Embedding: {row['sentence1_embeddings'].shape}")
    #     print(f"Sentence 2 Embedding: {row['sentence2_embeddings'].shape}")
    #     print("\n")

    df.to_csv(f"{file[:-4]}_cleaned.csv")
