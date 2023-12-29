from typing import Any

import pandas as pd
import numpy as np
import torch

from ContradictDetectNN import get_sentence_embedding, count_negations
from persitent_homology import persistent_homology_features
from tqdm import tqdm
import csv

from webchat import name_conversion, client, TEXT_MODEL


def label_mapping(
    df: pd.DataFrame, from_col: str = "gold_label", to_col: str = "label", str_to_int: bool = True
) -> pd.DataFrame:
    """
    Map natural language labels in a dataset to integers
    :param df: Dataframe containing the labels that needs to be mapped
    :param from_col: The column containing the original labels
    :param to_col: The new column to save integer labels to
    :param str_to_int: The direction to map. When True, converts sting labels to integer ones. When False, integer labels are converted to string labels.
    :return: The dataframe with an extra column added for the new labels
    """
    if str_to_int:
        mapping = {
            "neutral": 0,
            "entailment": 1,
            "contradiction": 2,
        }
    else:
        mapping = {0: "neutral", 1: "entailment", 2: "contradiction"}
    df[to_col] = df[from_col].map(mapping)
    return df


def encode_sentence(language_model, tokenizer, sentence, device):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = language_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def create_subset_with_ratio(input_df, subset_percentage, label_column):
    # Calculate the size of the desired subset
    total_count = input_df[label_column].count()
    subset_size = int(subset_percentage * total_count)

    # Create an empty DataFrame to store the subset
    subset_df = pd.DataFrame(columns=input_df.columns)

    # Iterate through the unique labels and sample data based on the proportion of the original count
    unique_labels = input_df[label_column].unique()
    for label in unique_labels:
        if label != "-":
            label_subset_count = int((input_df[label_column] == label).sum() * subset_percentage)
            label_subset = input_df[input_df[label_column] == label].sample(label_subset_count)
            subset_df = pd.concat([subset_df, label_subset], ignore_index=True)

    # Shuffle the subset DataFrame to randomize the order
    subset_df = subset_df.sample(frac=1).reset_index(drop=True)

    return subset_df


def embed_and_ph(df_for_cleaning: pd.DataFrame, output_csv_path: str) -> None:
    """
    Given a dataframe, retrieve the data's BERT embeddings and PH features for dimensions 0 and 1, then save it. If the
    output file already exists, pick up where it left off
    :param df_for_cleaning: df containing data
    :param output_csv_path: The path to the output file
    :return: None
    """
    # Check if the output CSV file already exists
    try:
        with open(output_csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            # Count the number of rows in the existing CSV file
            existing_records = sum(1 for _ in reader) - 1  # Subtract 1 for the header
    except FileNotFoundError:
        existing_records = 0

    print(f"\tWriting to:  {output_csv_path}")
    print(f"Starting at Record: {existing_records + 1}")

    for i, row in tqdm(df_for_cleaning.iterrows(), total=len(df_for_cleaning)):
        if i >= existing_records:
            if row["gold_label"].lower() == "contradiction":
                label = 2
            elif row["gold_label"].lower() == "entailment":
                label = 1
            else:
                label = 0

            # Apply the get_sentence_embedding function to generate embeddings
            if "sentence1_embeddings" not in row:
                row["sentence1_embeddings"] = get_sentence_embedding(row["sentence1"])

            if "sentence2_embeddings" not in row:
                row["sentence2_embeddings"] = get_sentence_embedding(row["sentence2"])

            # Calculate negation count
            if "negation" not in row:
                row["negation"] = count_negations([str(row["sentence1"]).strip(), str(row["sentence2"]).strip()])

            if output_csv_path[-6:-4] == "ph":
                if "sentence1_ph_a" not in row:
                    s1_ph_features = persistent_homology_features([row["sentence1"].strip()])
                    row["sentence1_ph_a"] = s1_ph_features[0][0]
                    row["sentence1_ph_b"] = s1_ph_features[0][1]

                if "sentence2_ph_a" not in row:
                    s2_ph_features = persistent_homology_features([row["sentence2"].strip()])
                    row["sentence2_ph_a"] = s2_ph_features[0][0]
                    row["sentence2_ph_b"] = s2_ph_features[0][1]

                # Create a new row for the result DataFrame
                result_row = {
                    "gold_label": row["gold_label"].lower().strip(),
                    "sentence1": row["sentence1"].strip(),
                    "sentence2": row["sentence2"].strip(),
                    "label": label,
                    "sentence1_embeddings": row["sentence1_embeddings"],
                    "sentence2_embeddings": row["sentence2_embeddings"],
                    "sentence1_ph_a": row["sentence1_ph_a"],
                    "sentence1_ph_b": row["sentence1_ph_b"],
                    "sentence2_ph_a": row["sentence2_ph_a"],
                    "sentence2_ph_b": row["sentence2_ph_b"],
                    "negation": row["negation"],
                }
            else:
                # Create a new row for the result DataFrame
                result_row = {
                    "gold_label": row["gold_label"].strip(),
                    "sentence1": str(row["sentence1"]).strip(),
                    "sentence2": str(row["sentence2"]).strip(),
                    "label": label,
                    # "sentence1_embeddings": row["sentence1_embeddings"],
                    # "sentence2_embeddings": row["sentence2_embeddings"],
                    "negation": row["negation"],
                }

            with open(output_csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=result_row.keys())
                if i == 0:
                    writer.writeheader()  # Write the header only for the first row
                try:
                    writer.writerow(result_row)
                except UnicodeError:
                    print(row["sentence1"])
                    print(row["sentence2"])
                    raise UnicodeError


def fact_rephrase(phrase: str, namespace: str, subject: str) -> list[str]:
    """
    Given a sentence, break it up into individual facts.
    :param phrase: A phrase containing multiple facts to be distilled into separate ones
    :return: The split-up factual statements
    """
    msgs: list[dict] = [
        {
            "role": "system",
            "content": f"You are {name_conversion(to_snake=False, to_convert=namespace)}. "
                       f"Split up the sentences about {subject} I provide you into facts. "
                       "Each fact should be able to stand on it's own. "
                       "Tell me each fact on a new line. "
                       "Do not include anything in your response other than the facts. "
                       "Refer to everyone in the third person, including yourself",
        }
    ]
    prompt: str = f"Split this phrase into facts: {phrase}"
    msgs.append({"role": "user", "content": prompt})  # build current history of conversation for model

    res: Any = client.chat.completions.create(model=TEXT_MODEL, messages=msgs, temperature=0)  # conversation with LLM
    facts: str = str(res.choices[0].message.content).strip()  # get model response
    return [fact.strip() for fact in facts.split("\n")]


if __name__ == "__main__":
    # TO_CLEAN: list = [
    #     # "Data/SemEval2014T1/train_cleaned.csv",
    #     # "Data/SemEval2014T1/test_cleaned.csv",
    #     # "Data/SemEval2014T1/valid_cleaned.csv",
    #     # "Data/mismatch_cleaned.csv",
    #     # "Data/match_cleaned.csv",
    #     "Data/SNLI/valid_cleaned.csv",
    #     "Data/SNLI/test_cleaned.csv",
    #     "Data/SNLI/train_subset_cleaned.csv",
    #     # "Data/MultiNLI/train.csv",
    #     # "Data/MultiNLI/test_match.csv",
    #     # "Data/MultiNLI/test_mismatch.csv",
    # ]
    # for file in TO_CLEAN:
    #     print(f"\tIn Progress: {file}")
    #     df = pd.read_csv(file)
    #     embed_and_ph(df, f"{file[:-4]}_cleaned.csv" if file[-11:-4] != "cleaned" else f"{file[:-4]}_ph.csv")
    with open("Text Summaries/Summaries/john_pebble.txt") as char_file:
        for line in char_file.readlines():
            print(line.strip())
            print(fact_rephrase(line, "john_pebble", "John Pebble"), '\n')
