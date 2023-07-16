import logging
import os
import random
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

from BiLSTM import BiLSTMModel
from convert_dataset import read_npz_file
from variables import (
    BATCH_SIZE,
    DEVICE,
    DATASET,
    INPUT_SIZE,
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CHKPT_INTERVAL,
)

# Disable the logging level for the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)


def get_most_recent_file(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None

    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if not files:
        return None

    return files[0]


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


def create_train_dev_sets(data: list, dev_ratio: float) -> Tuple[List[int], List[int]]:
    """
    Create a dev set and training set from a list of data.

    :param data: The list of data to split.
    :param dev_ratio: The ratio of data to be allocated for the dev set.
    :return: A tuple containing the indexes of the dev set and training set.
    """
    data_size = len(data)
    indices = list(range(data_size))
    random.shuffle(indices)  # Shuffle the indices randomly

    dev_size = int(data_size * dev_ratio)  # Calculate the dev set size

    dev_indices = indices[:dev_size]  # Extract dev set indices from the beginning of the shuffled indices
    train_indices = indices[dev_size:]  # Extract training set indices from the remaining shuffled indices

    return train_indices, dev_indices


def create_batches(data: torch.Tensor) -> List[torch.Tensor]:
    # Split the data into batches
    return torch.split(data, BATCH_SIZE, dim=0)


def get_bert_embeddings(sentence1: str, sentence2: str) -> torch.Tensor:
    # Load pre-trained BERT model and tokenizer
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model: BertModel = BertModel.from_pretrained("bert-base-uncased")

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
        bert_outputs: Tuple[torch.Tensor] = bert_model(input_ids, attention_mask=attention_mask)
        embeddings: torch.Tensor = bert_outputs.last_hidden_state  # Extract the last hidden state

    return embeddings


def count_files(directory: str) -> int:
    """
    Count the number of files in a directory.

    :param directory: The path to the directory.
    :return: The number of files in the directory.
    """
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    return file_count


def train_model(
    model_number: int,
    training_dataloader,
    x_validation,
    y_validation,
    input_size,
    hidden_size,
    num_layers,
    output_size,
    epochs: int,
    learning_rate: float,
    chkpt_interval: int,
    device: str,
    path_to_load_model: str = "",
):
    # Initialize the BiLSTM model
    bilstm: BiLSTMModel = BiLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    # Define loss function and optimizer
    loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(bilstm.parameters(), lr=learning_rate)

    # Initialize lists for training and validation loss
    training_losses: List[float] = []
    validation_losses: List[float] = []

    # Load the pre-trained weights if available
    if path_to_load_model != "":
        checkpoint = torch.load(get_most_recent_file(path_to_load_model))

        # Load the pre-trained weights
        bilstm.load_state_dict(checkpoint["model_state_dict"])

        # Load the optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load the loss and validation losses
        training_losses = checkpoint["train_loss"]
        validation_losses = checkpoint["val_loss"]

        # Get the last epoch to resume training from that point
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    if start_epoch >= epochs:
        epochs += start_epoch
        # Training loop
        for epoch in range(start_epoch, epochs):
            bilstm.train()  # get to training
            train_loss_sum: float = 0.0
            train_samples: int = 0

            # Iterate over the shuffled batches
            for batch_x, batch_y in training_dataloader:
                # Forward pass
                predictions: torch.Tensor = bilstm(batch_x.to(device))
                # Create a tensor of zeros with the same number of samples
                zeros_tensor: torch.Tensor = torch.zeros(predictions.shape[0], 1).to(device)

                # Concatenate the predictions tensor with the zeros tensor
                predictions: torch.Tensor = torch.cat((predictions, zeros_tensor), dim=1).to(device)
                batch_y: torch.Tensor = batch_y.squeeze().to(device)

                # Calculate the training loss
                loss: torch.Tensor = loss_function(predictions, batch_y)
                train_loss_sum += loss.item() * len(batch_x)
                train_samples += len(batch_x)

                # Backpropagation
                loss.backward()
                optimizer.step()

            training_losses.append(train_loss_sum / train_samples)  # record avg training loss

            # Validation phase
            bilstm.eval()
            val_loss_sum: float = 0.0
            val_samples: int = 0
            with torch.no_grad():
                # Forward pass on the validation set
                val_predictions: torch.Tensor = bilstm(x_validation.to(device))
                # Create a tensor of zeros with the same number of samples
                zeros_tensor: torch.Tensor = torch.zeros(val_predictions.shape[0], 1).to(device)
                # Concatenate the predictions tensor with the zeros tensor
                val_predictions: torch.Tensor = torch.cat((val_predictions, zeros_tensor), dim=1).to(device)
                # Calculate the validation loss
                val_loss: torch.Tensor = loss_function(val_predictions, y_validation.squeeze().to(device))
                val_loss_sum += val_loss.item() * len(x_validation)
                val_samples += len(x_validation)

            validation_losses.append(val_loss_sum / val_samples)  # record avg validation loss

            # Print training and validation loss for each epoch
            print(
                f"Epoch {epoch + 1}/{epochs}: Train Loss: {training_losses[-1]:.4f}, Val Loss: {validation_losses[-1]:.4f}"
            )

            # Save checkpoint if we've reached the interval
            if (epoch + 1) % chkpt_interval == 0:
                # Define the checkpoint file path
                checkpoint_path = f"Checkpoint/{model_number}/checkpoint_{epoch + 1}.pth"

                # Save the checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": bilstm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": training_losses,
                    "val_loss": validation_losses,
                }
                torch.save(checkpoint, checkpoint_path)
    else:
        # Training loop
        for epoch in range(start_epoch, epochs):
            bilstm.train()  # get to training
            train_loss_sum: float = 0.0
            train_samples: int = 0

            # Iterate over the shuffled batches
            for batch_x, batch_y in training_dataloader:
                # Forward pass
                predictions: torch.Tensor = bilstm(batch_x.to(device))
                # Create a tensor of zeros with the same number of samples
                zeros_tensor: torch.Tensor = torch.zeros(predictions.shape[0], 1).to(device)

                # Concatenate the predictions tensor with the zeros tensor
                predictions: torch.Tensor = torch.cat((predictions, zeros_tensor), dim=1).to(device)
                batch_y: torch.Tensor = batch_y.squeeze().to(device)

                # Calculate the training loss
                loss: torch.Tensor = loss_function(predictions, batch_y)
                train_loss_sum += loss.item() * len(batch_x)
                train_samples += len(batch_x)

                # Backpropagation
                loss.backward()
                optimizer.step()

            training_losses.append(train_loss_sum / train_samples)  # record avg training loss

            # Validation phase
            bilstm.eval()
            val_loss_sum: float = 0.0
            val_samples: int = 0
            with torch.no_grad():
                # Forward pass on the validation set
                val_predictions: torch.Tensor = bilstm(x_validation.to(device))
                # Create a tensor of zeros with the same number of samples
                zeros_tensor: torch.Tensor = torch.zeros(val_predictions.shape[0], 1).to(device)
                # Concatenate the predictions tensor with the zeros tensor
                val_predictions: torch.Tensor = torch.cat((val_predictions, zeros_tensor), dim=1).to(device)
                # Calculate the validation loss
                val_loss: torch.Tensor = loss_function(val_predictions, y_validation.squeeze().to(device))
                val_loss_sum += val_loss.item() * len(x_validation)
                val_samples += len(x_validation)

            validation_losses.append(val_loss_sum / val_samples)  # record avg validation loss

            # Print training and validation loss for each epoch
            print(
                f"Epoch {epoch + 1}/{EPOCHS}: Train Loss: {training_losses[-1]:.4f}, Val Loss: {validation_losses[-1]:.4f}"
            )

            # Save checkpoint if we've reached the interval
            if (epoch + 1) % chkpt_interval == 0:
                # Define the checkpoint file path
                checkpoint_path = f"Checkpoint/{model_number}/checkpoint_{epoch + 1}.pth"

                # Save the checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": bilstm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": training_losses,
                    "val_loss": validation_losses,
                }
                torch.save(checkpoint, checkpoint_path)
    return bilstm, training_losses, validation_losses, epochs


def clean_train_data(dataset: str, device: str, batch_size: int) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    multinli_df: pd.DataFrame = load_txt_file_to_dataframe(dataset)  # get data from file for labels
    file_data = read_npz_file(f"Data/MultiNLI/{dataset}_embeddings.npz")  # Read embeddings of data generated by BERT
    data_3d = np.stack(list(file_data.values()), axis=0)  # reformat data as 3d array
    data_2d = data_3d.reshape(data_3d.shape[0], -1)

    # Make labels
    labels: List[int] = [1 if x == "contradiction" else 0 for x in multinli_df["gold_label"]]

    # Create training and dev sets
    x_train, x_validation, y_train, y_validation = train_test_split(data_2d, labels, test_size=0.2)
    # Convert x_train and x_val to tensors
    x_train: torch.Tensor = (
        torch.tensor(x_train).to(device).reshape(x_train.shape[0], data_3d.shape[1], data_3d.shape[2])
    )
    x_validation: torch.Tensor = torch.tensor(x_validation).to(device).reshape(x_validation.shape[0], data_3d.shape[1], data_3d.shape[2])

    # Convert y_train and y_val to tensors
    y_train: torch.Tensor = torch.tensor(y_train).to(device)
    y_validation: torch.Tensor = torch.tensor(y_validation).to(device)

    # Convert training data and labels to TensorDataset
    train_dataset: TensorDataset = TensorDataset(x_train, y_train)

    # Create a DataLoader with shuffled data
    training_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return training_dataloader, x_validation, y_validation


if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(DEVICE))
    else:
        print("GPU is not available. PyTorch is using CPU.")

    # Create the "Models" folder if it doesn't exist
    if not os.path.exists("Models"):
        os.makedirs("Models")
    # Create the "Models" folder if it doesn't exist
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    model_num: int = count_files("Models")

    # Create the "Checkpoint" folder if it doesn't exist
    if not os.path.exists(f"Checkpoint/{model_num}"):
        os.makedirs(f"Checkpoint/{model_num}")

    model_save_path: str = f"Models/model{model_num}.pth"
    model_load_path: str = f"Checkpoint/{model_num - 1}"

    train_dataloader, x_val, y_val = clean_train_data(DATASET, DEVICE, BATCH_SIZE)

    model, train_losses, val_losses, x_range = train_model(
        model_num,
        train_dataloader,
        x_val,
        y_val,
        INPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        OUTPUT_SIZE,
        EPOCHS,
        LEARNING_RATE,
        CHKPT_INTERVAL,
        DEVICE,
        path_to_load_model=model_load_path,
    )

    # Save the entire model
    torch.save(model, model_save_path)

    # Plot the training and validation loss curves
    plt.plot(range(1, x_range + 1), train_losses, label="Training Loss")
    plt.plot(range(1, x_range + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"Figures/fig_{model_num}.svg", format="svg")
    plt.close()
