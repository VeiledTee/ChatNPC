import concurrent.futures
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


# Disable the logging level for the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define hyperparameters
SEQUENCE_LENGTH: int = 128
DATASET: str = "train"
if DATASET == "train":
    BATCH_SIZE: int = 1024
else:
    BATCH_SIZE: int = 128

# Check if GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def load_txt_file_to_dataframe(dataset_description: str) -> pd.DataFrame:
    """
    Load MultiNLI data into dataframe for use
    :param dataset_description: Which dataset to load and work with
    :return: Cleaned data contained in a dataframe
    """
    to_drop: list = [
        "label1",
        "sentence1_binary_parse",
        "sentence2_binary_parse",
        "sentence1_parse",
        "sentence2_parse",
        "promptID",
        "pairID",
        "genre",
        "label2",
        "label3",
        "label4",
        "label5",
    ]
    if dataset_description.lower().strip() == "train":
        data_frame = pd.read_csv("Data/MultiNLI/multinli_1.0_train.txt", sep="\t", encoding="latin-1").drop(
            columns=to_drop
        )
    elif dataset_description.lower().strip() == "match":
        data_frame = pd.read_csv("Data/MultiNLI/multinli_1.0_dev_matched.txt", sep="\t").drop(columns=to_drop)
    elif dataset_description.lower().strip() == "mismatch":
        data_frame = pd.read_csv("Data/MultiNLI/multinli_1.0_dev_mismatched.txt", sep="\t").drop(columns=to_drop)
    else:
        raise ValueError("Pass only 'train', 'match', or 'mismatch' to the function")

    data_frame.dropna(inplace=True)
    return data_frame


def parallel_get_bert_embeddings(batches) -> np.ndarray:
    embeddings: list = []
    start: int = get_start()
    print(f"Starting at batch: {start}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        total_batches: int = len(batches)
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for i, batch in enumerate(batches):
                if i >= start:
                    futures: list = []
                    ids: List[int] = batch[3]
                    for b in range(len(batch[0])):
                        future = executor.submit(get_bert_embeddings, batch[0][b], batch[1][b])
                        futures.append(future)
                    results = np.array([future.result() for future in futures]).squeeze()
                    embeddings.extend(results)

                    # Save embeddings from each batch with corresponding IDs as keys
                    batch_output_file = f"Data/NPZ/{DATASET.capitalize()}/batch_{i}.npz"
                    np.savez_compressed(
                        batch_output_file, **{str(identity): emb for identity, emb in zip(ids, results)}
                    )
                pbar.update(1)

    return np.array(embeddings)


def get_bert_embeddings(sentence1: str, sentence2: str) -> np.ndarray:
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Move model to the specified device
    model.to(DEVICE)

    # Tokenize the sentences and obtain the input IDs and attention masks
    encoding = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, padding="longest", truncation=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # Pad or truncate the input IDs and attention masks to the maximum sequence length
    input_ids = torch.nn.functional.pad(input_ids, (0, SEQUENCE_LENGTH - input_ids.size(1)))
    attention_mask = torch.nn.functional.pad(attention_mask, (0, SEQUENCE_LENGTH - attention_mask.size(1)))

    # Obtain the BERT embeddings and convert to NumPy array on CPU
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.cpu().numpy()

    return embeddings


def group_rows(dataframe):
    grouped_data = []
    for i in range(0, len(dataframe), BATCH_SIZE):
        batch = dataframe.iloc[i : i + BATCH_SIZE]
        sentenceA: List[str] = [x for x in batch["sentence1"]]
        sentenceB: List[str] = [x for x in batch["sentence2"]]
        labels: List[int] = [1 if x == "contradiction" else 0 for x in batch["gold_label"]]
        ids: List[int] = list(batch.index)
        grouped_data.append([sentenceA, sentenceB, labels, ids])
    return grouped_data


def combine_npz_files(output_file: str, npz_dir: str = f"Data/NPZ/{DATASET.capitalize()}"):
    npz_files = [file for file in os.listdir(npz_dir) if file.endswith(".npz")]
    combined_data = {}
    for file in npz_files:
        file_data = read_npz_file(f"{npz_dir}/{file}")
        for key, value in file_data.items():
            combined_data[key] = value
    np.savez_compressed(output_file, **combined_data)


def read_npz_file(file_path):
    loaded_data = np.load(file_path)
    data = {}

    for key in loaded_data.files:
        data[key] = loaded_data[key]

    return data


def delete_npz_files(npz_dir: str = f"Data/NPZ/{DATASET.capitalize()}"):
    for file in os.listdir(npz_dir):
        if file.endswith(".npz"):
            file_path = os.path.join(npz_dir, file)
            os.remove(file_path)


def get_start(npz_dir: str = f"Data/NPZ/{DATASET.capitalize()}") -> int:
    npz_files = [file for file in os.listdir(npz_dir) if file.endswith(".npz")]  # retrieve npz files
    return len(npz_files)  # return largest completed batch


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(DEVICE))
    else:
        print("GPU is not available. PyTorch is using CPU.")
    # combine_npz_files()
    # retrieved = read_npz_file("Data/MultiNLI/match_embeddings.npz")
    # for k, array in retrieved.items():
    #     print(f"Array with key '{k}': {array.shape}")

    for directory in ["Data/NPZ", "Data/MultiNLI", f"Data/NPZ/{DATASET.capitalize()}"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    multinli_df: pd.DataFrame = load_txt_file_to_dataframe(DATASET)
    print(f"\nRow count: {len(multinli_df)}")
    data_batches = group_rows(multinli_df)
    print(f"Num Batches: {len(data_batches)}")
    embedded_batches = parallel_get_bert_embeddings(data_batches)
    print(f"Embedding count: {len(embedded_batches)}")
    print(f"Embedding shape: {embedded_batches.shape}")
    combine_npz_files(output_file=f"Data/MultiNLI/{DATASET}_embeddings.npz")
    retrieved = read_npz_file(f"Data/MultiNLI/{DATASET}_embeddings.npz")
    for k, array in retrieved.items():
        if not np.array_equal(embedded_batches[int(k)], array):
            print(f"Array {k} not equal")
    # delete_npz_files()
