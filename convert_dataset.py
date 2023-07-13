import concurrent.futures
import logging
import queue
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# Disable the logging level for the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define hyperparameters
SEQUENCE_LENGTH: int = 128
BATCH_SIZE: int = 128

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
        data_frame = pd.read_csv("Data/MultiNLI/multinli_1.0_dev_matched.txt", sep="\t", nrows=10).drop(columns=to_drop)
    elif dataset_description.lower().strip() == "mismatch":
        data_frame = pd.read_csv("Data/MultiNLI/multinli_1.0_dev_mismatched.txt", sep="\t").drop(columns=to_drop)
    else:
        raise ValueError("Pass only 'train', 'match', or 'mismatch' to the function")

    data_frame.dropna(inplace=True)
    return data_frame


def split_into_batches(df: pd.DataFrame) -> List[List[List[str | int]]]:
    num_batches = len(df) // BATCH_SIZE
    batches = []

    gold_label: List[int] = [1 if x == "contradiction" else 0 for x in df["gold_label"]]

    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE

        batch: Tuple[List[List[str]], List[List[str]], List[int], List[int]] = (
            df["sentence1"].iloc[start_index:end_index].tolist(),
            df["sentence2"].iloc[start_index:end_index].tolist(),
            gold_label[start_index:end_index],
            df.index[start_index:end_index].tolist(),
        )

        batches.append(batch)

    # Handle the remaining rows if the dataframe size is not divisible by BATCH_SIZE
    if len(df) % BATCH_SIZE != 0:
        start_index = num_batches * BATCH_SIZE
        batch: Tuple[List[List[str]], List[List[str]], List[int], List[int]] = (
            df["sentence1"].iloc[start_index:].tolist(),
            df["sentence2"].iloc[start_index:].tolist(),
            gold_label[start_index:],
            df.index[start_index:].tolist(),
        )

        batches.append(batch)

    return batches


def save_embeddings(arr: np.ndarray, IDs: np.ndarray, filename: str) -> None:
    # Check if the file already exists
    try:
        existing_data = np.load(filename)

        # If the file exists, extract the existing data
        existing_arr = existing_data["arr"]
        existing_ids = existing_data["IDs"]

        # Reshape the existing_arr to have the same dimensions as arr
        existing_arr = np.reshape(existing_arr, (existing_arr.shape[0],) + arr.shape[1:])

        # Append the new data to the existing arrays
        arr = np.concatenate((existing_arr, arr))
        existing_ids += IDs[:existing_ids.shape[0]]
        IDs = existing_ids
    except FileNotFoundError:
        pass

    # Split the array into individual arrays
    split_arrays = np.vsplit(arr, arr.shape[0])

    # Save each split array with its corresponding ID
    np.savez_compressed(filename, arr=split_arrays, IDs=IDs)


def get_bert_embeddings(sentence1: str, sentence2: str) -> np.ndarray:
    # Load pre-trained BERT model and tokenizer
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model: BertModel = BertModel.from_pretrained("bert-base-uncased")

    # Tokenize the sentences and obtain the input IDs and attention masks
    tokens: Dict[str, torch.Tensor] = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, padding="longest", truncation=True
    )
    input_ids: torch.Tensor = torch.tensor(tokens["input_ids"]).unsqueeze(0)  # Add batch dimension
    attention_mask: torch.Tensor = torch.tensor(tokens["attention_mask"]).unsqueeze(0)  # Add batch dimension

    # Pad or truncate the input IDs and attention masks to the maximum sequence length
    input_ids = torch.nn.functional.pad(input_ids, (0, SEQUENCE_LENGTH - input_ids.size(1)))
    attention_mask = torch.nn.functional.pad(attention_mask, (0, SEQUENCE_LENGTH - attention_mask.size(1)))

    # Obtain the BERT embeddings
    with torch.no_grad():
        bert_outputs: Tuple[torch.Tensor] = model(input_ids, attention_mask=attention_mask)
        embeddings: torch.Tensor = bert_outputs.last_hidden_state  # Extract the last hidden state

    return embeddings.numpy()


def process_batch(batch):
    sentenceA, sentenceB = batch[0], batch[1]
    x_train = []

    q = queue.Queue(maxsize=16)  # Set the maximum number of concurrent threads to 16

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
        futures = []

        for i in range(len(sentenceA)):
            # Submit the task to the executor and add the future object to the list
            futures.append(exe.submit(get_bert_embeddings, sentenceA[i], sentenceB[i]))

            # Check if the maximum number of concurrent threads is reached
            if len(futures) >= 16:
                # Wait for the first completed future and retrieve the result
                completed = concurrent.futures.as_completed(futures)
                for future in completed:
                    x_train.append(future.result())  # Append the result to the list
                    futures.remove(future)  # Remove the completed future

        # Process the remaining futures
        for future in concurrent.futures.as_completed(futures):
            x_train.append(future.result())  # Append the result to the list

    arrays = np.array([np.squeeze(x) for x in x_train])
    return arrays


if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(device))
    else:
        print("GPU is not available. PyTorch is using CPU.")

    # load data
    multinli_df: pd.DataFrame = load_txt_file_to_dataframe("train")  # all train
    # multinli_df: pd.DataFrame = load_txt_file_to_dataframe('match')  # 10 rows of match
    # multinli_df: pd.DataFrame = load_txt_file_to_dataframe('mismatch')  # all mismatch
    print(len(multinli_df.index))
    data_batches = split_into_batches(multinli_df)
    print(len(data_batches))
    embeddings = []

    with tqdm(total=len(data_batches)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_batch, data_batches)
            for result, batch in zip(results, data_batches):
                embeddings.append(result)
                pbar.update(1)
                save_embeddings(result, np.array(batch[3]), "Data/MultiNLI/training.npz")

    # embeddings = np.concatenate(embeddings)
