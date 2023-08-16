import pandas as pd
import numpy as np

from ContradictDetectNN import get_sentence_embedding, count_negations
from persitent_homology import persistent_homology_features


def embed_and_ph(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["gold_label"].str.lower()
    df["label"] = np.where(df["label"] == "contradiction", 1, 0)
    print("labels complete")
    # Apply the get_sentence_embedding function to generate embeddings
    if "sentence1_embeddings" in df.columns:
        print("sentence1_embeddings already exists")
    else:
        df["sentence1_embeddings"] = df["sentence1"].apply(lambda x: get_sentence_embedding(x))
        print("sentence1_embeddings complete")
    s1_ph_features = persistent_homology_features(list(df["sentence1"]))
    df["sentence1_ph_a"] = [item[0] for item in s1_ph_features]
    df["sentence1_ph_b"] = [item[1] for item in s1_ph_features]
    print("sentence1_features complete")
    if "sentence2_embeddings" in df.columns:
        print("sentence2_embeddings already exists")
    else:
        df["sentence2_embeddings"] = df["sentence2"].apply(lambda x: get_sentence_embedding(x))
        print("sentence2_embeddings complete")
    s2_ph_features = persistent_homology_features(list(df["sentence2"]))
    df["sentence2_ph_a"] = [item[0] for item in s2_ph_features]
    df["sentence2_ph_b"] = [item[1] for item in s2_ph_features]
    print("sentence2_features complete")
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
    return df


if __name__ == "__main__":
    TO_CLEAN: list = ["Data/match_cleaned.csv", "Data/mismatch_cleaned.csv"]
    for file in TO_CLEAN:
        print(f"\tIn Progress: {file}")
        df = pd.read_csv(file)
        df = embed_and_ph(df)
        df.to_csv(f"{file[:-4]}_ph.csv", index=False)
