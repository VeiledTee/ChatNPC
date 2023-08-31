from typing import Dict, Tuple, List, Any

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch.nn.functional as F

from variables import DEVICE
import logging
from persitent_homology import persistent_homology_features
import matplotlib.pyplot as plt

logging.getLogger("transformers").setLevel(logging.ERROR)


# SEED = 42
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


def analyze_float_list(data: List[float]) -> None:
    """
    Calculate and print various statistical measures from a list of floats.
    :param data: A list of floats to analyze
    :return: None
    """
    data_array: np.ndarray = np.array(data)

    mean: float = np.mean(data_array)
    median: float = np.median(data_array)
    maximum: float = max(data_array)
    minimum: float = min(data_array)
    std_deviation: float = np.std(data_array)
    variance: float = np.var(data_array)
    skewness: float = stats.skew(data_array)
    kurtosis: float = stats.kurtosis(data_array)
    percentile_25: float = np.percentile(data_array, 25)
    percentile_75: float = np.percentile(data_array, 75)

    print(f"Mean:                 {mean:.2f}")
    print(f"Median:               {median:.2f}")
    print(f"Minimum:              {minimum:.2f}")
    print(f"Maximum:              {maximum:.2f}")
    print(f"Standard Deviation:   {std_deviation:.2f}")
    print(f"Variance:             {variance:.2f}")
    print(f"Skewness:             {skewness:.2f}")
    print(f"Kurtosis:             {kurtosis:.2f}")
    print(f"25th Percentile:      {percentile_25:.2f}")
    print(f"75th Percentile:      {percentile_75:.2f}")


def apply_get_bert_embeddings(row):
    return get_bert_embeddings(row["sentence1"], row["sentence2"]).squeeze()


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
    input_ids = torch.nn.functional.pad(input_ids, (0, 50 - input_ids.size(1)))
    attention_mask = torch.nn.functional.pad(attention_mask, (0, 50 - attention_mask.size(1)))

    # Obtain the BERT embeddings
    with torch.no_grad():
        bert_outputs: Tuple[torch.Tensor] = bert_model(input_ids, attention_mask=attention_mask)
        embeddings: torch.Tensor = bert_outputs.last_hidden_state  # Extract the last hidden state
    print(embeddings.shape)
    return embeddings


def get_sentence_embedding(sentence: str) -> torch.Tensor:
    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(DEVICE)

    # Tokenize the input sentence
    tokens = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    tokens = {key: value.to(DEVICE) for key, value in tokens.items()}

    # Get the model output
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the representation of [CLS] token (sentence embedding)
    sentence_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

    # Move the sentence embedding tensor to CPU before returning
    return sentence_embedding.squeeze().cpu()


def count_negations(sentences: List[str]) -> int:
    negation_count: int = 0
    for sentence in sentences:
        # Load the BERT tokenizer
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Count the negation words
        negation_words: List[str] = ["not", "no", "never", "none", "nobody", "nowhere", "nothing", "neither", "nor"]
        # Preprocess contractions
        sentence: str = sentence.replace("n't", " not")
        sentence: str = sentence.replace("'re", " are")
        sentence: str = sentence.replace("'ll", " will")
        # tokenize
        words: tokenizer = tokenizer.tokenize(sentence)
        # Count the negation words
        negation_count += sum(1 for word in words if word.lower() in negation_words)

    return negation_count


def str_to_tensor(string: Any) -> torch.Tensor:
    string = string.replace("\n", "")
    nums = string[9:-2].strip().split(", ")
    num_list = [float(num_str) for num_str in nums]
    return torch.tensor(num_list)


def map_words_to_glove_embeddings(sentence, embeddings, max_length: int = 64):
    words = sentence.split()
    padding_length = max_length - len(words)
    sentence_embedding = np.mean([embeddings[word] for word in words if word in embeddings], axis=0)
    padded_sentence_embedding = np.pad(sentence_embedding, [(0, padding_length), (0, 0)], mode="constant")
    return torch.tensor(padded_sentence_embedding)


def plot_training_history(
    train_accuracy_list: list, val_accuracy_list: list, train_f1_list: list, val_f1_list: list, number_of_epochs: int
):
    # Create x-axis values (epochs)
    x_axis: list = list(range(1, number_of_epochs + 1))

    # Create separate plots for accuracy and F1-score
    plt.figure(figsize=(12, 4))

    # Training accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, train_accuracy_list, label="Training Accuracy", marker="o")
    plt.plot(x_axis, val_accuracy_list, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # Training F1-score plot
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, train_f1_list, label="Training F1 Score", marker="o")
    plt.plot(x_axis, val_f1_list, label="Validation F1 Score", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


class ConvBBU(nn.Module):
    def __init__(self):
        super(ConvBBU, self).__init__()
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, 1)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2 = inputs
        # Convolutional
        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)
        # TanH
        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)
        # Reshape
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # Calculate difference
        concatenated = x2 - x1
        # Feed to forward composition layers
        x = self.fc1(concatenated)
        x = self.dropout1(x)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(str_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(str_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(str_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence1_embeddings"]), dim=0
        )
        sentence2_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence2_embeddings"]), dim=0
        )

        device = torch.device(device)
        self.to(device)
        criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: np.ndarray = training_data["label"].iloc[i : i + batch_size].values
                batch_labels: torch.Tensor = (
                    torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                # Forward pass
                # outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations, batch_s1_feature_a, batch_s1_feature_b, batch_s2_feature_a, batch_s2_feature_b])
                outputs: torch.Tensor = model([s1_embedding, s2_embedding])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to binary predictions (0 or 1)
                predicted_labels: np.ndarray = (outputs >= 0.5).float().view(-1).cpu().numpy()
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()
                all_predicted_labels.extend(predicted_labels)
                all_true_labels.extend(true_labels)
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels)

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: np.ndarray = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)

                    # Forward pass for validation
                    val_outputs: torch.Tensor = model([s1_embedding, s2_embedding])

                    # Convert validation outputs to binary predictions (0 or 1)
                    val_predicted_labels: np.ndarray = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
                    all_val_predicted_labels.extend(val_predicted_labels)

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels)
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(str_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                output: torch.Tensor = model([s1_embedding, s2_embedding])
                predicted_labels: np.ndarray = (output >= 0.5).float().cpu().numpy()
                final_predictions = np.append(final_predictions, predicted_labels)
            return final_predictions


class ConvBBUNeg(nn.Module):
    def __init__(self):
        super(ConvBBUNeg, self).__init__()
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, 1)  # output layer

    def forward(self, inputs: tuple):
        x1, x2, num_negation = inputs

        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)

        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Subtract the representations of x1 and x2
        concatenated = x2 - x1
        x = self.fc1(concatenated)
        x = self.dropout1(x)

        # Add num_negation to x
        x = torch.cat((x, num_negation), dim=1)

        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(str_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(str_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(str_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_training_embeddings = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings = torch.stack(list(validation_data["sentence1_embeddings"]), dim=0)
        sentence2_validation_embeddings = torch.stack(list(validation_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)
        criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: np.ndarray = training_data["label"].iloc[i : i + batch_size].values
                batch_labels: torch.Tensor = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1)
                # Get additional feature values
                num_negations: np.ndarray = training_data["negation"].iloc[i : i + batch_size].values
                batch_negations: torch.Tensor = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(
                    -1, 1
                )

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                batch_labels: torch.Tensor = batch_labels.to(device)
                batch_negations: torch.Tensor = batch_negations.to(device)
                # Forward pass
                outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to binary predictions (0 or 1)
                predicted_labels: np.ndarray = (outputs >= 0.5).float().view(-1).cpu().numpy()
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()
                all_predicted_labels.extend(predicted_labels)
                all_true_labels.extend(true_labels)
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels)

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: np.ndarray = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]
                    batch_negations = validation_data["negation"].iloc[i : i + batch_size].values

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)
                    batch_negations: torch.Tensor = (
                        torch.tensor(batch_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                    )

                    # Forward pass for validation
                    val_outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])

                    # Convert validation outputs to binary predictions (0 or 1)
                    val_predicted_labels: np.ndarray = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
                    all_val_predicted_labels.extend(val_predicted_labels)

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels)
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(str_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                # Get additional feature values
                num_negations: np.ndarray = test_data["negation"].iloc[i : i + batch_size].values
                batch_negations: torch.Tensor = (
                    torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )
                output: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
                predicted_labels: np.ndarray = (output >= 0.5).float().cpu().numpy()
                final_predictions = np.append(final_predictions, predicted_labels)
            return final_predictions


# Add PH
class ConvBBUPH(nn.Module):
    def __init__(self):
        super(ConvBBUPH, self).__init__()
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, 1)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2 = inputs
        # Convolutional
        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)
        # TanH
        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)
        # Reshape
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # Calculate difference
        concatenated = x2 - x1
        # Feed to forward composition layers
        x = self.fc1(concatenated)
        x = self.dropout1(x)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(str_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(str_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(str_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence1_embeddings"]), dim=0
        )
        sentence2_validation_embeddings: torch.Tensor = torch.stack(
            list(validation_data["sentence2_embeddings"]), dim=0
        )

        device = torch.device(device)
        self.to(device)
        criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: np.ndarray = training_data["label"].iloc[i : i + batch_size].values
                batch_labels: torch.Tensor = (
                    torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                # Forward pass
                # outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations, batch_s1_feature_a, batch_s1_feature_b, batch_s2_feature_a, batch_s2_feature_b])
                outputs: torch.Tensor = model([s1_embedding, s2_embedding])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to binary predictions (0 or 1)
                predicted_labels: np.ndarray = (outputs >= 0.5).float().view(-1).cpu().numpy()
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()
                all_predicted_labels.extend(predicted_labels)
                all_true_labels.extend(true_labels)
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels)

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: np.ndarray = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)

                    # Forward pass for validation
                    val_outputs: torch.Tensor = model([s1_embedding, s2_embedding])

                    # Convert validation outputs to binary predictions (0 or 1)
                    val_predicted_labels: np.ndarray = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
                    all_val_predicted_labels.extend(val_predicted_labels)

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels)
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(str_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                output: torch.Tensor = model([s1_embedding, s2_embedding])
                predicted_labels: np.ndarray = (output >= 0.5).float().cpu().numpy()
                final_predictions = np.append(final_predictions, predicted_labels)
            return final_predictions


# Add PH
class ConvBBUNegPH(nn.Module):
    def __init__(self):
        super(ConvBBUNegPH, self).__init__()
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, 1)  # output layer

    def forward(self, inputs: tuple):
        x1, x2, num_negation = inputs

        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)

        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Subtract the representations of x1 and x2
        concatenated = x2 - x1
        x = self.fc1(concatenated)
        x = self.dropout1(x)

        # Add num_negation to x
        x = torch.cat((x, num_negation), dim=1)

        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(str_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(str_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(str_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_training_embeddings = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings = torch.stack(list(validation_data["sentence1_embeddings"]), dim=0)
        sentence2_validation_embeddings = torch.stack(list(validation_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)
        criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_predicted_labels: list = []
            all_true_labels: list = []
            for i in range(0, len(training_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: np.ndarray = training_data["label"].iloc[i : i + batch_size].values
                batch_labels: torch.Tensor = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1)
                # Get additional feature values
                num_negations: np.ndarray = training_data["negation"].iloc[i : i + batch_size].values
                batch_negations: torch.Tensor = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(
                    -1, 1
                )

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                batch_labels: torch.Tensor = batch_labels.to(device)
                batch_negations: torch.Tensor = batch_negations.to(device)
                # Forward pass
                outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to binary predictions (0 or 1)
                predicted_labels: np.ndarray = (outputs >= 0.5).float().view(-1).cpu().numpy()
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()
                all_predicted_labels.extend(predicted_labels)
                all_true_labels.extend(true_labels)
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels)

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: np.ndarray = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]
                    batch_negations = validation_data["negation"].iloc[i : i + batch_size].values

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)
                    batch_negations: torch.Tensor = (
                        torch.tensor(batch_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                    )

                    # Forward pass for validation
                    val_outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])

                    # Convert validation outputs to binary predictions (0 or 1)
                    val_predicted_labels: np.ndarray = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
                    all_val_predicted_labels.extend(val_predicted_labels)

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels)
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(str_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(str_to_tensor)
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                # Get additional feature values
                num_negations: np.ndarray = test_data["negation"].iloc[i : i + batch_size].values
                batch_negations: torch.Tensor = (
                    torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )
                output: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
                predicted_labels: np.ndarray = (output >= 0.5).float().cpu().numpy()
                final_predictions = np.append(final_predictions, predicted_labels)
            return final_predictions


if __name__ == "__main__":
    NUM_EPOCHS = 10
    BATCH_SIZE = 64

    # Load and preprocess the data
    # n = 80
    # v = 10
    # t = 10
    n = None
    v = None
    t = None

    # Initialize empty lists to store training and validation metrics
    tests_acc: list = []
    tests_f1: list = []

    # for _ in range(30):  # stat significance testing
    #     # load model
    #     model = ConvBBU()
    #     device = torch.device(DEVICE)
    #     model = model.to(device)
    #     criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    #     # initialize data containers for plotting
    #     train_accuracy_values: list = []
    #     train_f1_values: list = []
    #     val_accuracy_values: list = []
    #     val_f1_values: list = []
    #
    #     print(f"Model on {device}")
    #
    #     for epoch in range(NUM_EPOCHS):
    #         model.train()  # Set the model to training mode
    #         running_loss: float = 0.0
    #         all_predicted_labels: list = []
    #         all_true_labels: list = []
    #
    #         for i in range(0, len(train_df), BATCH_SIZE):
    #             # Prepare the batch
    #             s1_embedding: torch.Tensor = train_bert_embeddings_sentence1[i : i + BATCH_SIZE]
    #             s2_embedding: torch.Tensor = train_bert_embeddings_sentence2[i : i + BATCH_SIZE]
    #             # Get the corresponding labels for this batch
    #             batch_labels = train_df["label"].iloc[i : i + BATCH_SIZE].values
    #             batch_labels: torch.Tensor = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1)
    #             # Get additional feature values
    #             num_negations = train_df["negation"].iloc[i : i + BATCH_SIZE].values
    #             batch_negations = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1)
    #
    #             # s1_ph_features = persistent_homology_features(list(train_df["sentence1"].iloc[i : i + BATCH_SIZE]))
    #             # dim_0_s1_features = [item[0] for item in s1_ph_features]
    #             # # dim_1_s1_features = [item[1] for item in s1_ph_features]
    #             # batch_s1_feature_a = torch.tensor(np.array(dim_0_s1_features)).to(device)
    #             # # batch_s1_feature_b = torch.tensor(np.array(dim_1_s1_features)).to(device)
    #             # print(batch_s1_feature_a.shape)
    #             # print(batch_s1_feature_b.shape)
    #
    #             # s2_ph_features = persistent_homology_features(list(train_df["sentence2"].iloc[i : i + BATCH_SIZE]))
    #             # dim_0_s2_features = [item[0] for item in s2_ph_features]
    #             # # dim_1_s2_features = [item[1] for item in s2_ph_features]
    #             # batch_s2_feature_a = torch.tensor(np.array(dim_0_s2_features)).to(device)
    #             # # batch_s2_feature_b = torch.tensor(np.array(dim_1_s2_features)).to(device)
    #             # print(batch_s2_feature_a.shape)
    #             # print(batch_s2_feature_b.shape)
    #
    #             # Move tensors to the device
    #             s1_embedding: torch.Tensor = s1_embedding.to(device)
    #             s2_embedding: torch.Tensor = s2_embedding.to(device)
    #             batch_labels = batch_labels.to(device)
    #             batch_negations = batch_negations.to(device)
    #             # Forward pass
    #             # outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations, batch_s1_feature_a, batch_s1_feature_b, batch_s2_feature_a, batch_s2_feature_b])
    #             outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
    #             # Compute the loss
    #             loss = criterion(outputs, batch_labels)
    #             # Backpropagation
    #             optimizer.zero_grad()  # Clear accumulated gradients
    #             loss.backward()
    #             # Optimize (update model parameters)
    #             optimizer.step()
    #             # Update running loss
    #             running_loss += loss.item()
    #             # Convert outputs to binary predictions (0 or 1)
    #             predicted_labels: np.ndarray = (outputs >= 0.5).float().view(-1).cpu().numpy()
    #             true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()
    #             all_predicted_labels.extend(predicted_labels)
    #             all_true_labels.extend(true_labels)
    #         # Calculate training accuracy and F1-score
    #         train_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    #         train_f1 = f1_score(all_true_labels, all_predicted_labels)
    #
    #         # Print training metrics for this epoch
    #         average_loss: float = running_loss / (len(train_df) / BATCH_SIZE)
    #
    #         # Validation
    #         model.eval()  # Set the model to evaluation mode
    #         all_val_predicted_labels: np.ndarray = []
    #
    #         with torch.no_grad():
    #             for i in range(0, len(valid_df), BATCH_SIZE):
    #                 # Prepare the batch for validation
    #                 s1_embedding: torch.Tensor = valid_bert_embeddings_sentence1[i : i + BATCH_SIZE]
    #                 s2_embedding: torch.Tensor = valid_bert_embeddings_sentence2[i : i + BATCH_SIZE]
    #                 batch_negations = valid_df["negation"].iloc[i : i + BATCH_SIZE].values
    #
    #                 # Move tensors to the device
    #                 s1_embedding: torch.Tensor = s1_embedding.to(device)
    #                 s2_embedding: torch.Tensor = s2_embedding.to(device)
    #                 batch_negations = (
    #                     torch.tensor(batch_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
    #                 )
    #
    #                 # Forward pass for validation
    #                 val_outputs: torch.Tensor = model([s1_embedding, s2_embedding, batch_negations])
    #
    #                 # Convert validation outputs to binary predictions (0 or 1)
    #                 val_predicted_labels: np.ndarray = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
    #                 all_val_predicted_labels.extend(val_predicted_labels)
    #
    #         # Calculate validation accuracy and F1-score
    #         val_accuracy: float = accuracy_score(valid_df["label"], all_val_predicted_labels)
    #         val_f1: float = f1_score(valid_df["label"], all_val_predicted_labels)
    #         # print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    #         # print(f"\tTraining   | Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, Loss: {average_loss:.4f}")
    #         # print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
    #         train_accuracy_values.append(train_accuracy)
    #         train_f1_values.append(train_f1)
    #         val_accuracy_values.append(val_accuracy)
    #         val_f1_values.append(val_f1)
    #
    #     # plot_training_history(train_accuracy_values, val_accuracy_values, train_f1_values, val_f1_values, NUM_EPOCHS)
    #
    #     # model_save_path: str = "Models/test1.pt"
    #     # torch.save(model.state_dict(), model_save_path)
    #     # model.load_state_dict(torch.load(model_save_path))
    #
    #     with torch.no_grad():
    #         model.eval()  # Set the model to evaluation mode
    #         predictions = np.array([])
    #         for i in range(0, len(test_df), BATCH_SIZE):
    #             # Prepare the batch
    #             s1_embedding: torch.Tensor = test_bert_embeddings_sentence1[i : i + BATCH_SIZE].to(device)
    #             s2_embedding: torch.Tensor = test_bert_embeddings_sentence2[i : i + BATCH_SIZE].to(device)
    #             # Get additional feature values
    #             num_negations = test_df["negation"].iloc[i : i + BATCH_SIZE].values
    #             batch_negations = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
    #             output = model([s1_embedding, s2_embedding, batch_negations])
    #             predicted_labels: np.ndarray = (output >= 0.5).float().cpu().numpy()
    #             predictions = np.append(predictions, predicted_labels)
    #
    #         true_labels: np.ndarray = test_df["label"].values
    #         train_accuracy = accuracy_score(true_labels, predictions)
    #         train_f1 = f1_score(true_labels, predictions)
    #         tests_acc.append(train_accuracy)
    #         tests_f1.append(train_f1)
    #         print(f"Test Accuracy: {train_accuracy:.4f}, Test F1 Score: {train_f1:.4f}")
    #         print(f"Percent of positive class: {sum(true_labels) / len(true_labels):.4f}%")
    for name, model in [("SoloBERT", ConvBBU()), ("NegationBERT", ConvBBUNeg())]:
        if n is not None and v is not None and t is not None:
            train_df = pd.read_csv("Data/match_cleaned.csv").head(n)
            valid_df = pd.read_csv("Data/mismatch_cleaned.csv").head(v)
            test_df = pd.read_csv("Data/contradiction-dataset_cleaned_ph.csv").head(t)
        else:
            # train_df = pd.concat([
            #     # pd.read_csv("Data/match_cleaned.csv"),
            #     pd.read_csv("Data/mismatch_cleaned.csv"),
            #     # pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv"),
            #     # pd.read_csv("Data/SemEval2014T1/valid_cleaned.csv"),
            # ], ignore_index=True)
            # valid_df = pd.read_csv("Data/contradiction-dataset_cleaned_ph.csv")
            # test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned.csv")
            # train_df = pd.read_csv("Data/match_cleaned.csv")
            # valid_df = pd.read_csv("Data/mismatch_cleaned.csv")
            # test_df = pd.read_csv("Data/contradiction-dataset_cleaned_ph.csv")
            train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
            valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned.csv")
            test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned.csv")
        train_accuracy, train_f1, valid_accuracy, valid_f1 = model.train_model(
            train_df, valid_df, BATCH_SIZE, NUM_EPOCHS, DEVICE, verbose=False
        )
        predictions = model.predict(test_df, BATCH_SIZE, DEVICE)
        print(f"==========\n{name}")
        print(f"Accuracy: {accuracy_score(test_df['label'].values, predictions):.4f}")
        print(f"F1-Score: {f1_score(test_df['label'].values, predictions):.4f}")
    # print(f"Avg acc over {len(tests_acc)} runs: {sum(tests_acc) / len(tests_acc)}")
    # print(f"Best Accuracy:        {max(tests_acc)}")
    # print(f"Worst Accuracy:       {min(tests_acc)}")
    # print(f"Avg f1 over {len(tests_acc)} runs:  {sum(tests_f1) / len(tests_f1)}")
    # print(f"Best F1 Score:        {max(tests_f1)}")
    # print(f"Worst F1 Score:       {min(tests_f1)}")
    """
    match-mismatch-contradiction
    Avg acc over 30 runs: 0.7149188514357054
    Best Accuracy:        0.7640449438202247
    Worst Accuracy:       0.649812734082397
    Avg f1 over 30 runs:  0.8275341820334495
    Best F1 Score:        0.8630434782608696
    Worst F1 Score:       0.7823050058207216
    """
    """
    semeval train-valid-test
    """
    # print(f"=====\n\tAccuracy Breakdown\n=====")
    # analyze_float_list(tests_acc)
    # print(f"=====\n\tF1 Score Breakdown\n=====")
    # analyze_float_list(tests_f1)
    """
    match-mismatch-contradiction 1
    =====
        Accuracy Breakdown
    =====
    Mean:                 0.70
    Median:               0.71
    Minimum:              0.47
    Maximum:              0.81
    Standard Deviation:   0.04
    Variance:             0.00
    Skewness:             -0.76
    Kurtosis:             1.19
    25th Percentile:      0.67
    75th Percentile:      0.73
    =====
        F1 Score Breakdown
    =====
    Mean:                 0.82
    Median:               0.82
    Minimum:              0.63
    Maximum:              0.89
    Standard Deviation:   0.03
    Variance:             0.00
    Skewness:             -1.01
    Kurtosis:             2.01
    25th Percentile:      0.80
    75th Percentile:      0.84
    """
    """
    match-mismatch-contradiction Aug 30 7:30am
    =====
        Accuracy Breakdown
    =====
    Mean:                 0.70
    Median:               0.71
    Minimum:              0.57
    Maximum:              0.78
    Standard Deviation:   0.05
    Variance:             0.00
    Skewness:             -0.92
    Kurtosis:             1.28
    25th Percentile:      0.69
    75th Percentile:      0.72
    =====
        F1 Score Breakdown
    =====
    Mean:                 0.82
    Median:               0.82
    Minimum:              0.71
    Maximum:              0.88
    Standard Deviation:   0.03
    Variance:             0.00
    Skewness:             -1.19
    Kurtosis:             1.90
    25th Percentile:      0.81
    75th Percentile:      0.83
    """
    """
    semeval train-semeval valid-semeval test
    =====
        Accuracy Breakdown
    =====
    Mean:                 0.86
    Median:               0.86
    Minimum:              0.85
    Maximum:              0.86
    Standard Deviation:   0.00
    Variance:             0.00
    Skewness:             0.57
    Kurtosis:             -0.47
    25th Percentile:      0.85
    75th Percentile:      0.86
    =====
        F1 Score Breakdown
    =====
    Mean:                 0.09
    Median:               0.08
    Minimum:              0.00
    Maximum:              0.24
    Standard Deviation:   0.06
    Variance:             0.00
    Skewness:             0.77
    Kurtosis:             -0.01
    25th Percentile:      0.04
    75th Percentile:      0.12
    """
    # NUM_EPOCHS = 10
    # BATCH_SIZE = 64
    #
    # # glove_embeddings = {}  # Store GloVe embeddings
    # # with open('Data/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    # #     for line in f:
    # #         values = line.strip().split()
    # #         word = values[0]
    # #         vector = np.array(values[1:], dtype='float32')
    # #         glove_embeddings[word] = vector
    #
    # # Load and preprocess the data
    # n = None
    # v = None
    # t = None
    # if n is not None and v is not None and t is not None:
    #     train_df = pd.read_csv("Data/match_cleaned.csv").head(n)
    #     valid_df = pd.read_csv("Data/mismatch_cleaned.csv").head(v)
    #     test_df = pd.read_csv("Data/contradiction-dataset_cleaned.csv").head(t)
    # else:
    #     train_df = pd.read_csv("Data/match_cleaned.csv")
    #     valid_df = pd.read_csv("Data/mismatch_cleaned.csv")
    #     test_df = pd.read_csv("Data/contradiction-dataset_cleaned.csv")
    #
    # # Load and preprocess the embeddings
    # train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(str_to_tensor)
    # train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(str_to_tensor)
    # valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(str_to_tensor)
    # valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(str_to_tensor)
    # test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(str_to_tensor)
    # test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(str_to_tensor)
    #
    # # Stack tensors to pass to model
    # train_bert_embeddings_sentence1 = torch.stack(list(train_df["sentence1_embeddings"]), dim=0)
    # train_bert_embeddings_sentence2 = torch.stack(list(train_df["sentence2_embeddings"]), dim=0)
    # valid_bert_embeddings_sentence1 = torch.stack(list(valid_df["sentence1_embeddings"]), dim=0)
    # valid_bert_embeddings_sentence2 = torch.stack(list(valid_df["sentence2_embeddings"]), dim=0)
    # test_bert_embeddings_sentence1 = torch.stack(list(test_df["sentence1_embeddings"]), dim=0)
    # test_bert_embeddings_sentence2 = torch.stack(list(test_df["sentence2_embeddings"]), dim=0)
    #
    # # train_glove_embeddings_sentence1 = torch.stack(
    # #     [torch.tensor(map_words_to_glove_embeddings(sentence, glove_embeddings)) for sentence in train_df["sentence1"]]
    # # )
    # # train_glove_embeddings_sentence2 = torch.stack(
    # #     [torch.tensor(map_words_to_glove_embeddings(sentence, glove_embeddings)) for sentence in train_df["sentence2"]]
    # # )
    # # valid_glove_embeddings_sentence1 = torch.stack(
    # #     [torch.tensor(map_words_to_glove_embeddings(sentence, glove_embeddings)) for sentence in valid_df["sentence1"]]
    # # )
    # # valid_glove_embeddings_sentence2 = torch.stack(
    # #     [torch.tensor(map_words_to_glove_embeddings(sentence, glove_embeddings)) for sentence in valid_df["sentence2"]]
    # # )
    # # test_glove_embeddings_sentence1 = torch.stack(
    # #     [torch.tensor(map_words_to_glove_embeddings(sentence, glove_embeddings)) for sentence in test_df["sentence1"]]
    # # )
    # # test_glove_embeddings_sentence2 = torch.stack(
    # #     [torch.tensor(map_words_to_glove_embeddings(sentence, glove_embeddings)) for sentence in test_df["sentence2"]]
    # # )
    #
    # # Load Siamese model
    # siamese_model = SentenceSiameseClassifier()
    # device = torch.device(DEVICE)
    # siamese_model = siamese_model.to(device)
    # criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(device)
    # optimizer = optim.Adam(siamese_model.parameters(), lr=0.1)
    #
    # print(f"Model on {device}")
    #
    # for epoch in range(NUM_EPOCHS):
    #     siamese_model.train()  # Set the model to training mode
    #     running_loss: float = 0.0
    #     all_true_labels: list = []
    #     all_predicted_labels: list = []
    #
    #     for i in range(0, len(train_df), BATCH_SIZE):
    #         # Prepare the batch
    #         s1_embedding: torch.Tensor = train_bert_embeddings_sentence1[i: i + BATCH_SIZE].to(device)
    #         s2_embedding: torch.Tensor = train_bert_embeddings_sentence2[i: i + BATCH_SIZE].to(device)
    #         # Get the corresponding labels for this batch
    #         batch_labels = train_df["label"].iloc[i: i + BATCH_SIZE].values
    #         batch_labels: torch.Tensor = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1).to(device)
    #         # Get additional feature values
    #         num_negations = train_df["negation"].iloc[i: i + BATCH_SIZE].values
    #         batch_negations = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
    #
    #         # Forward pass
    #         outputs = siamese_model([s1_embedding, s2_embedding, batch_negations])
    #         # Compute the loss
    #         loss = criterion(outputs, batch_labels)
    #         # Backpropagation
    #         optimizer.zero_grad()  # Clear accumulated gradients
    #         loss.backward()
    #         # Optimize (update model parameters)
    #         optimizer.step()
    #         # Update running loss
    #         running_loss += loss.item()
    #         # Store true labels for later evaluation
    #         predicted_labels: np.ndarray = (outputs >= 0.5).float().view(-1).cpu().numpy()
    #         true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()
    #         all_true_labels.extend(true_labels)
    #         all_predicted_labels.extend(predicted_labels)
    #
    #     # Calculate training accuracy and F1-score
    #     accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    #     f1 = f1_score(all_true_labels, all_predicted_labels)
    #
    #     # Print training metrics for this epoch
    #     average_loss: float = running_loss / (len(train_df) / BATCH_SIZE)
    #     print(
    #         f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}"
    #     )
    #
    #     # Validation
    #     siamese_model.eval()  # Set the model to evaluation mode
    #     all_val_predicted_labels: np.ndarray = []
    #
    #     with torch.no_grad():
    #         for i in range(0, len(valid_df), BATCH_SIZE):
    #             # Prepare the batch for validation
    #             s1_embedding: torch.Tensor = valid_bert_embeddings_sentence1[i: i + BATCH_SIZE].to(device)
    #             s2_embedding: torch.Tensor = valid_bert_embeddings_sentence2[i: i + BATCH_SIZE].to(device)
    #             batch_negations = (
    #                 torch.tensor(valid_df["negation"].iloc[i: i + BATCH_SIZE].values, dtype=torch.float32)
    #                 .view(-1, 1)
    #                 .to(device)
    #             )
    #
    #             # Forward pass for validation
    #             val_outputs = siamese_model([s1_embedding, s2_embedding, batch_negations])
    #
    #             # Convert validation outputs to binary predictions (0 or 1)
    #             val_predicted_labels: np.ndarray = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
    #             all_val_predicted_labels.extend(val_predicted_labels)
    #
    #     # Calculate validation accuracy and F1-score
    #     val_accuracy: float = accuracy_score(valid_df["label"], all_val_predicted_labels)
    #     val_f1: float = f1_score(valid_df["label"], all_val_predicted_labels)
    #
    #     print(f"\tValidation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}")
    #
    # # Testing
    # siamese_model.eval()  # Set the model to evaluation mode
    # predictions = np.array([])
    #
    # with torch.no_grad():
    #     for i in range(0, len(test_df), BATCH_SIZE):
    #         # Prepare the batch
    #         s1_embedding: torch.Tensor = test_bert_embeddings_sentence1[i: i + BATCH_SIZE].to(device)
    #         s2_embedding: torch.Tensor = test_bert_embeddings_sentence2[i: i + BATCH_SIZE].to(device)
    #         batch_negations = (
    #             torch.tensor(test_df["negation"].iloc[i: i + BATCH_SIZE].values, dtype=torch.float32)
    #             .view(-1, 1)
    #             .to(device)
    #         )
    #         output = siamese_model([s1_embedding, s2_embedding, batch_negations])
    #         predicted_labels: np.ndarray = (output >= 0.5).float().cpu().numpy()
    #         predictions = np.append(predictions, predicted_labels)
    #
    # true_labels: np.ndarray = test_df["label"].values
    # accuracy = accuracy_score(true_labels, predictions)
    # f1 = f1_score(true_labels, predictions)
    #
    # print(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")
