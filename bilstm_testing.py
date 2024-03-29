import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from BiLSTM import BiLSTMModel
from bilstm_training import load_txt_file_to_dataframe
from convert_dataset import read_npz_file, get_bert_embeddings
from config import (
    BATCH_SIZE,
    DEVICE,
    TESTSET,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
)

# Disable the logging level for the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)


def clean_test_data(dataset: str, device: str, batch_size: int) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    multinli_df: pd.DataFrame = load_txt_file_to_dataframe(dataset)  # get data from file for labels
    file_data = read_npz_file(f"Data/MultiNLI/{dataset}_embeddings.npz")  # Read embeddings of data generated by BERT
    data_3d = np.stack(list(file_data.values()), axis=0)  # reformat data as 3d array
    data_2d = data_3d.reshape(data_3d.shape[0], -1)

    # Make labels
    labels: List[int] = [1 if x == "contradiction" else 0 for x in multinli_df["gold_label"]]

    # Convert data and labels to tensors
    x_data: torch.Tensor = (
        torch.tensor(data_2d).to(device).reshape(data_2d.shape[0], data_3d.shape[1], data_3d.shape[2])
    )
    y_labels: torch.Tensor = torch.tensor(labels).to(device)

    # Create dataset and dataloader
    dataset: TensorDataset = TensorDataset(x_data, y_labels)
    dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, x_data, y_labels


if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(DEVICE))
    else:
        print("GPU is not available. PyTorch is using CPU.")

    # DEVICE = 'cpu'

    model_load_path: str = f"Models/train_match.pth"
    test_sentences = [
        ["Billy loves cake", "Billy hates cake"],
        ["The sky is clear today.", "What a beautiful sunny day!"],
        ["Josh doesn't like pizza.", "Josh loves eating pizza!"],
    ]
    test_labels = [1, 0, 1]

    if test_sentences:
        test_x = torch.tensor(np.array([get_bert_embeddings(s[0], s[1]) for s in test_sentences])).squeeze()
        test_y = torch.tensor(test_labels)
    else:
        test_dataloader, test_x, test_y = clean_test_data(TESTSET, DEVICE, BATCH_SIZE)

    model = BiLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(model_load_path, map_location=DEVICE).state_dict())
    model.eval()

    with torch.no_grad():
        output = model(test_x.to(DEVICE))
        max_pooling_output, _ = torch.max(output, dim=1)
        print(max_pooling_output)
        avg_pooling_output = torch.mean(output, dim=1)
        print(avg_pooling_output)
        probabilities = torch.softmax(output, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        output_np = predicted_labels.cpu().numpy()

    print(f"Actual:      {np.array(test_y)}")
    print(f"Predictions: {output_np}")

    print(f"Actual sum:     {sum(np.array(test_y))}")
    print(f"Prediction sum: {sum(output_np)}")

    accuracy = np.mean(np.array(test_y) == output_np)
    score = accuracy.item()

    print(f"Score: {score}")
