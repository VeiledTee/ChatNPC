from typing import Tuple, Dict, List
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import logging
from torch.utils.data import DataLoader, TensorDataset
import os

from tqdm import tqdm
import concurrent.futures

from BiLSTM import BiLSTMModel
import pandas as pd
import math


# Define hyperparameters
INPUT_SIZE: int = 768
SEQUENCE_LENGTH: int = 128
HIDDEN_SIZE: int = 64
NUM_LAYERS: int = 2
OUTPUT_SIZE: int = 1
EPOCHS: int = 250
BATCH_SIZE: int = 10
LEARNING_RATE: float = 0.001
CHKPT_INTERVAL: int = int(math.ceil(EPOCHS / 10))


def load_txt_file_to_dataframe(dataset_description: str) -> pd.DataFrame:
    """
    Load MultiNLI data into dataframe for use
    :param dataset_description: Which dataset to load and work with
    :return: Cleaned data contained in a dataframe
    """
    to_drop: list = ['label1', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'promptID', 'pairID', 'genre', 'label2', 'label3', 'label4', 'label5']
    if dataset_description.lower().strip() == 'train':
        data_frame = pd.read_csv('Data/MultiNLI/multinli_1.0_train.txt', sep='\t').drop(columns=to_drop)
    elif dataset_description.lower().strip() == 'match':
        data_frame = pd.read_csv('Data/MultiNLI/multinli_1.0_dev_matched.txt', sep='\t', nrows=10).drop(columns=to_drop)
    elif dataset_description.lower().strip() == 'mismatch':
        data_frame = pd.read_csv('Data/MultiNLI/multinli_1.0_dev_mismatched.txt', sep='\t').drop(columns=to_drop)
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
    # Disable the logging level for the transformers library
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Load pre-trained BERT model and tokenizer
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model: BertModel = BertModel.from_pretrained('bert-base-uncased')

    # Restoring the logging level
    logging.getLogger("transformers").setLevel(logging.INFO)

    # Tokenize the sentences and obtain the input IDs and attention masks
    tokens: Dict[str, torch.Tensor] = tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True,
                                                            padding='longest', truncation=True)
    input_ids: torch.Tensor = torch.tensor(tokens['input_ids']).unsqueeze(0)  # Add batch dimension
    attention_mask: torch.Tensor = torch.tensor(tokens['attention_mask']).unsqueeze(0)  # Add batch dimension

    # Pad or truncate the input IDs and attention masks to the maximum sequence length
    input_ids = torch.nn.functional.pad(input_ids, (0, SEQUENCE_LENGTH - input_ids.size(1)))
    attention_mask = torch.nn.functional.pad(attention_mask, (0, SEQUENCE_LENGTH - attention_mask.size(1)))

    # Obtain the BERT embeddings
    with torch.no_grad():
        bert_outputs: Tuple[torch.Tensor] = model(input_ids, attention_mask=attention_mask)
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


if __name__ == '__main__':
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(device))
    else:
        print("GPU is not available. PyTorch is using CPU.")

    # Create the "Checkpoint" folder if it doesn't exist
    if not os.path.exists('Checkpoint'):
        os.makedirs('Checkpoint')
    # Create the "Models" folder if it doesn't exist
    if not os.path.exists("Models"):
        os.makedirs("Models")
    # Create the "Models" folder if it doesn't exist
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    model_num: int = count_files('Models')

    # load data
    # multinli_df: pd.DataFrame = load_txt_file_to_dataframe('match')  # 10 rows
    multinli_df: pd.DataFrame = load_txt_file_to_dataframe('mismatch')  # all

    # Create train/validation sets
    training_indices, validation_indices = create_train_dev_sets(list(multinli_df['gold_label'].values), dev_ratio=0.2)

    # Two lists of sentences for training
    sentenceA: List[str] = [x for x in multinli_df['sentence1']]
    sentenceB: List[str] = [x for x in multinli_df['sentence2']]
    # print(f"A: {sentenceA}")
    # print(f"B: {sentenceB}")

    # Make labels
    y_train: List[int] = [1 if x == 'contradiction' else 0 for x in multinli_df['gold_label']]
    # print(f"L: {y_train}")

    x_train: list = []
    # Using ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Calculate the total number of iterations
        total_iterations: int = len(sentenceA)

        # Wrap the parallelized execution with tqdm
        with tqdm(total=total_iterations) as pbar:
            # Map the function to each pair of sentences in parallel
            results = executor.map(get_bert_embeddings, sentenceA, sentenceB)

            # Collect the results
            for result in results:
                x_train.append(result)

                # Update progress bar
                pbar.update(1)

    # Create training and dev sets
    training_x: torch.Tensor = torch.stack([x_train[i] for i in training_indices]).view(len(training_indices), 128, 768)  # reshape to 3d
    # print(f"X Train: {training_x.shape}")
    validation_x: torch.Tensor = torch.stack([x_train[i] for i in validation_indices]).view(len(validation_indices), 128, 768)  # reshape to 3d
    # print(f"X Val: {validation_x.shape}")
    training_y: torch.Tensor = torch.tensor([[y_train[i]] for i in training_indices], dtype=torch.long) # convert to float for loss function, same shape as predictions
    # print(f"Y Train: {training_y.shape}")
    # print(training_y)
    validation_y: torch.Tensor = torch.tensor([[y_train[i]] for i in validation_indices], dtype=torch.long)  # convert to float for loss function, same shape as predictions
    # print(f"Y Val: {validation_y.shape}")

    # Convert training data and labels to TensorDataset
    train_dataset: TensorDataset = TensorDataset(training_x, training_y)

    # Create a DataLoader with shuffled data
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the BiLSTM model
    model: BiLSTMModel = BiLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

    # Define loss function and optimizer
    loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize lists for training and validation loss
    train_losses: List[float] = []
    val_losses: List[float] = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()  # get to training
        train_loss_sum: float = 0.0
        train_samples: int = 0

        # Iterate over the shuffled batches
        for batch_x, batch_y in train_dataloader:
            # Forward pass
            predictions: torch.Tensor = model(batch_x.to(device))
            # Create a tensor of zeros with the same number of samples
            zeros_tensor: torch.Tensor = torch.zeros(predictions.shape[0], 1).to(device)

            # Concatenate the predictions tensor with the zeros tensor
            predictions: torch.Tensor = torch.cat((predictions, zeros_tensor), dim=1).to(device)
            batch_y: torch.Tensor = batch_y.squeeze().to(device)

            # Calculate the training loss
            loss: torch.Tensor = loss_function(predictions, batch_y)  # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
            train_loss_sum += loss.item() * len(batch_x)
            train_samples += len(batch_x)

            # Backpropagation
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss_sum / train_samples)  # record avg training loss

        # Validation phase
        model.eval()
        val_loss_sum: float = 0.0
        val_samples: int = 0
        with torch.no_grad():
            # Forward pass on the validation set
            val_predictions: torch.Tensor = model(validation_x.to(device))
            # Create a tensor of zeros with the same number of samples
            zeros_tensor: torch.Tensor = torch.zeros(val_predictions.shape[0], 1).to(device)
            # Concatenate the predictions tensor with the zeros tensor
            val_predictions: torch.Tensor = torch.cat((val_predictions, zeros_tensor), dim=1).to(device)
            # Calculate the validation loss
            val_loss: torch.Tensor = loss_function(val_predictions, validation_y.squeeze().to(device))
            val_loss_sum += val_loss.item() * len(validation_x)
            val_samples += len(validation_x)

        val_losses.append(val_loss_sum / val_samples)  # record avg validation loss

        # Print training and validation loss for each epoch
        print(f"Epoch {epoch + 1}/{EPOCHS}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        # Save checkpoint if we've reached the interval
        if (epoch + 1) % CHKPT_INTERVAL == 0:
            # Define the checkpoint file path
            checkpoint_path = f'Checkpoint/checkpoint{model_num}_{epoch + 1}.pth'

            # Save the checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1]
            }
            torch.save(checkpoint, checkpoint_path)

    # Save the entire model
    torch.save(model, f'Models/model{model_num}.pth')

    # Plot the training and validation loss curves
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"Figures/fig_{model_num}.svg", format='svg')
    plt.close()
