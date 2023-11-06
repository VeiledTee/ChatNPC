from typing import Dict, Tuple, List, Any

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Filter and ignore UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
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


def embedding_to_tensor(string: Any) -> torch.Tensor:
    string = string.replace("\n", "")
    nums = string[9:-2].strip().split(", ")
    num_list = [float(num_str) for num_str in nums]
    return torch.tensor(num_list)


def ph_to_tensor(array_str):
    # Remove newline characters and strip brackets
    array_str = array_str.replace("\n", "").strip("[]")

    # Split rows and convert to a list of lists of floats
    rows = [list(map(float, row.split())) for row in array_str.split("] [")]

    # Check for 'inf' values and replace them
    for row in rows:
        for i in range(len(row)):
            if np.isinf(row[i]):
                row[i] = np.finfo(np.float32).max  # Replace 'inf' with a large finite value

    # Convert the list of lists to a NumPy array
    array = np.array(rows)

    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(array, dtype=torch.float32)

    return tensor


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
    def __init__(self, num_classes: int):
        super(ConvBBU, self).__init__()
        self.num_classes = num_classes
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, num_classes)  # output layer with 'num_classes' units

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2 = inputs
        # Convolutional
        x1 = self.conv1(x1.unsqueeze(2))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = self.conv1(x2.unsqueeze(2))  # Add a channel dimension
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
        final_layer_output = self.fc2(x)  # Linear output, no activation

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
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
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []

            for i in range(0, len(training_data), batch_size):
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]

                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)

                # Forward pass
                outputs: torch.Tensor = self([s1_embedding, s2_embedding])

                # Compute the loss
                loss: float = criterion(outputs, batch_labels)

                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()

                # Optimize (update model parameters)
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())

            average_loss: float = running_loss / (len(training_data) / batch_size)

            # Calculate training accuracy and F1-score
            training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
            training_f1: float = f1_score(
                all_true_labels, all_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            train_accuracy_values.append(training_accuracy)
            train_f1_values.append(training_f1)

            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)

                    # Forward pass for validation
                    val_outputs: torch.Tensor = self([s1_embedding, s2_embedding])

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_true_labels = validation_data["label"].values
            val_accuracy: float = accuracy_score(val_true_labels, all_val_predicted_labels)
            val_f1: float = f1_score(
                val_true_labels, all_val_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)

        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)

                # Forward pass for predictions
                output: torch.Tensor = self([s1_embedding, s2_embedding])
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions


class ConvBBUNeg(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvBBUNeg, self).__init__()
        self.num_classes = num_classes
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, num_classes)  # output layer with 'num_classes' units

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

        final_layer_output = self.fc2(x)

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
        # Stack embeddings for batch processing
        sentence1_training_embeddings = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)
        sentence1_validation_embeddings = torch.stack(list(validation_data["sentence1_embeddings"]), dim=0)
        sentence2_validation_embeddings = torch.stack(list(validation_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
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
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Get additional feature values
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                batch_negations: np.ndarray = (
                    torch.tensor(training_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                    .view(-1, 1)
                    .to(device)
                )
                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                # Forward pass
                outputs: torch.Tensor = self([s1_embedding, s2_embedding, batch_negations])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average="macro")

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)
                    batch_negations: np.ndarray = (
                        torch.tensor(validation_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                        .view(-1, 1)
                        .to(device)
                    )

                    # Forward pass for validation
                    val_outputs: torch.Tensor = self([s1_embedding, s2_embedding, batch_negations])
                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)
                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_true_labels = validation_data["label"].values
            val_accuracy: float = accuracy_score(val_true_labels, all_val_predicted_labels)
            val_f1: float = f1_score(
                val_true_labels, all_val_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []
            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                # Get additional feature values
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                num_negations: np.ndarray = test_data["negation"].iloc[i : i + batch_size].values
                batch_negations: torch.Tensor = (
                    torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )

                output: torch.Tensor = self([s1_embedding, s2_embedding, batch_negations])
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                all_true_labels.extend(true_labels.cpu().numpy())
                final_predictions = np.append(final_predictions, predicted_classes.cpu().numpy())
            return final_predictions


# Add PH
class ConvBBUPH(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvBBUPH, self).__init__()
        # Define model layers
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(1186, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, self.num_classes)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2, ph1a, ph1b, ph2a, ph2b = inputs  # ph shape: [x, 260, 3] amd [x, 50, 3]
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
        x_concat = x2 - x1  # [x, 256]
        pha_concat = ph1a - ph2a  # [x, 260, 3]
        phb_concat = ph1b + ph2b  # [x, 50, 3]

        pha_concat_reshaped = pha_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 260*3]
        phb_concat_reshaped = phb_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 50*3]

        # Now you can concatenate x_concat, pha_concat_reshaped, and phb_concat_reshaped along dimension 1
        final_input = torch.cat(
            (x_concat, pha_concat_reshaped, phb_concat_reshaped), dim=1
        )  # [x, 256 + 260*3 + 50*3] ([x, 1186])

        # Feed to forward composition layers
        x = self.fc1(final_input)
        x = self.dropout1(x)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            training_data[column] = training_data[column].apply(ph_to_tensor)
            # Validation cleaning
            validation_data[column] = validation_data[column].apply(ph_to_tensor)

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
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
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
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = torch.Tensor(sentence1_training_embeddings[i : i + batch_size]).to(device)
                s2_embedding: torch.Tensor = torch.Tensor(sentence2_training_embeddings[i : i + batch_size]).to(device)
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)

                # Forward pass
                outputs: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average="macro")

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size].to(device)
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size].to(device)

                    # Prepare PH vectors
                    batch_s1_feature_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s1_feature_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)

                    # Forward pass
                    val_outputs: torch.Tensor = self(
                        [
                            s1_embedding,
                            s2_embedding,
                            batch_s1_feature_a,
                            batch_s1_feature_b,
                            batch_s2_feature_a,
                            batch_s2_feature_b,
                        ]
                    )

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels, average="macro")
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Test cleaning
            test_data[column] = test_data[column].apply(ph_to_tensor)

        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                output: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions


# Add PH
class ConvBBUNegPH(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvBBUNegPH, self).__init__()
        # Define model layers
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(1186, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, self.num_classes)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2, num_negation, ph1a, ph1b, ph2a, ph2b = inputs  # ph shape: [x, 260, 3] amd [x, 50, 3]
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
        x_concat = x2 - x1  # [x, 256]
        pha_concat = ph1a - ph2a  # [x, 260, 3]
        phb_concat = ph1b + ph2b  # [x, 50, 3]

        pha_concat_reshaped = pha_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 260*3]
        phb_concat_reshaped = phb_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 50*3]

        # Now you can concatenate x_concat, pha_concat_reshaped, and phb_concat_reshaped along dimension 1
        final_input = torch.cat(
            (x_concat, pha_concat_reshaped, phb_concat_reshaped), dim=1
        )  # [x, 256 + 260*3 + 50*3] ([x, 1186])
        # Feed to forward composition layers
        x = self.fc1(final_input)
        x = self.dropout1(x)
        # Add num_negation to x
        x = torch.cat((x, num_negation), dim=1)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # Clean data
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence1_embeddings"] = validation_data["sentence1_embeddings"].apply(embedding_to_tensor)
        validation_data["sentence2_embeddings"] = validation_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            training_data[column] = training_data[column].apply(ph_to_tensor)
            # Validation cleaning
            validation_data[column] = validation_data[column].apply(ph_to_tensor)

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
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
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
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = torch.Tensor(sentence1_training_embeddings[i : i + batch_size]).to(device)
                s2_embedding: torch.Tensor = torch.Tensor(sentence2_training_embeddings[i : i + batch_size]).to(device)
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)

                batch_negations: np.ndarray = (
                    torch.tensor(training_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                    .view(-1, 1)
                    .to(device)
                )

                # Forward pass
                outputs: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_negations,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average="macro")

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size].to(device)
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size].to(device)
                    batch_negations: np.ndarray = (
                        torch.tensor(validation_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                        .view(-1, 1)
                        .to(device)
                    )

                    # Prepare PH vectors
                    batch_s1_feature_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s1_feature_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)

                    # Forward pass
                    val_outputs: torch.Tensor = self(
                        [
                            s1_embedding,
                            s2_embedding,
                            batch_negations,
                            batch_s1_feature_a,
                            batch_s1_feature_b,
                            batch_s2_feature_a,
                            batch_s2_feature_b,
                        ]
                    )

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels, average="macro")
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Test cleaning
            test_data[column] = test_data[column].apply(ph_to_tensor)

        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_embeddings"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                batch_negations: np.ndarray = (
                    torch.tensor(test_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                    .view(-1, 1)
                    .to(device)
                )
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                output: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_negations,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions


class ConvSBERT(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERT, self).__init__()
        self.num_classes = num_classes
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, num_classes)  # output layer with 'num_classes' units

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2 = inputs
        # Convolutional
        x1 = self.conv1(x1.unsqueeze(2))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = self.conv1(x2.unsqueeze(2))  # Add a channel dimension
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
        final_layer_output = self.fc2(x)  # Linear output, no activation

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        self.embedding_model.to(device)
        # Clean data
        training_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence1"]
        ]
        training_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence2"]
        ]
        validation_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence1"]
        ]
        validation_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_sbert"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_sbert"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence1_sbert"]), dim=0)
        sentence2_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence2_sbert"]), dim=0)

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)

        # initialize data containers for plotting
        train_accuracy_values: list = []
        train_f1_values: list = []
        val_accuracy_values: list = []
        val_f1_values: list = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []

            for i in range(0, len(training_data), batch_size):
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]

                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)

                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)

                # Forward pass
                outputs: torch.Tensor = self([s1_embedding, s2_embedding])

                # Compute the loss
                loss: float = criterion(outputs, batch_labels)

                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()

                # Optimize (update model parameters)
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())

            average_loss: float = running_loss / (len(training_data) / batch_size)

            # Calculate training accuracy and F1-score
            training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
            training_f1: float = f1_score(
                all_true_labels, all_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            train_accuracy_values.append(training_accuracy)
            train_f1_values.append(training_f1)

            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)

                    # Forward pass for validation
                    val_outputs: torch.Tensor = self([s1_embedding, s2_embedding])

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_true_labels = validation_data["label"].values
            val_accuracy: float = accuracy_score(val_true_labels, all_val_predicted_labels)
            val_f1: float = f1_score(
                val_true_labels, all_val_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> ndarray:
        self.embedding_model.to(device)
        # Clean data
        test_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence1"]
        ]
        test_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_sbert"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_sbert"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)

                # Forward pass for predictions
                output: torch.Tensor = self([s1_embedding, s2_embedding])
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions


class ConvSBERTNeg(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERTNeg, self).__init__()
        self.num_classes = num_classes
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Define model layers
        self.conv1 = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, num_classes)  # output layer with 'num_classes' units

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

        final_layer_output = self.fc2(x)

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        self.embedding_model.to(device)
        # Clean data
        training_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence1"]
        ]
        training_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence2"]
        ]
        validation_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence1"]
        ]
        validation_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_sbert"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_sbert"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence1_sbert"]), dim=0)
        sentence2_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence2_sbert"]), dim=0)
        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
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
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_training_embeddings[i : i + batch_size]
                s2_embedding: torch.Tensor = sentence2_training_embeddings[i : i + batch_size]
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Get additional feature values
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                batch_negations: np.ndarray = (
                    torch.tensor(training_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                    .view(-1, 1)
                    .to(device)
                )
                # Move tensors to the device
                s1_embedding: torch.Tensor = s1_embedding.to(device)
                s2_embedding: torch.Tensor = s2_embedding.to(device)
                # Forward pass
                outputs: torch.Tensor = self([s1_embedding, s2_embedding, batch_negations])
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average="macro")

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size]
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size]

                    # Move tensors to the device
                    s1_embedding: torch.Tensor = s1_embedding.to(device)
                    s2_embedding: torch.Tensor = s2_embedding.to(device)
                    batch_negations: np.ndarray = (
                        torch.tensor(validation_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                        .view(-1, 1)
                        .to(device)
                    )

                    # Forward pass for validation
                    val_outputs: torch.Tensor = self([s1_embedding, s2_embedding, batch_negations])
                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)
                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_true_labels = validation_data["label"].values
            val_accuracy: float = accuracy_score(val_true_labels, all_val_predicted_labels)
            val_f1: float = f1_score(
                val_true_labels, all_val_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> ndarray:
        self.embedding_model.to(device)
        # Clean data
        test_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence1"]
        ]
        test_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_sbert"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_sbert"]), dim=0)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []
            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                # Get additional feature values
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                num_negations: np.ndarray = test_data["negation"].iloc[i : i + batch_size].values
                batch_negations: torch.Tensor = (
                    torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )

                output: torch.Tensor = self([s1_embedding, s2_embedding, batch_negations])
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                all_true_labels.extend(true_labels.cpu().numpy())
                final_predictions = np.append(final_predictions, predicted_classes.cpu().numpy())
            return final_predictions


# Add PH
class ConvSBERTPH(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERTPH, self).__init__()
        # Define model layers
        self.num_classes = num_classes
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.conv1 = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(1186, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, self.num_classes)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2, ph1a, ph1b, ph2a, ph2b = inputs  # ph shape: [x, 260, 3] amd [x, 50, 3]
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
        x_concat = x2 - x1  # [x, 256]
        pha_concat = ph1a - ph2a  # [x, 260, 3]
        phb_concat = ph1b + ph2b  # [x, 50, 3]

        pha_concat_reshaped = pha_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 260*3]
        phb_concat_reshaped = phb_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 50*3]

        # Now you can concatenate x_concat, pha_concat_reshaped, and phb_concat_reshaped along dimension 1
        final_input = torch.cat(
            (x_concat, pha_concat_reshaped, phb_concat_reshaped), dim=1
        )  # [x, 256 + 260*3 + 50*3] ([x, 1186])

        # Feed to forward composition layers
        x = self.fc1(final_input)
        x = self.dropout1(x)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        self.embedding_model.to(device)
        # Clean data
        training_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence1"]
        ]
        training_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence2"]
        ]
        validation_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence1"]
        ]
        validation_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_sbert"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_sbert"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence1_sbert"]), dim=0)
        sentence2_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence2_sbert"]), dim=0)

        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            training_data[column] = training_data[column].apply(ph_to_tensor)
            # Validation cleaning
            validation_data[column] = validation_data[column].apply(ph_to_tensor)

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
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
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = torch.Tensor(sentence1_training_embeddings[i : i + batch_size]).to(device)
                s2_embedding: torch.Tensor = torch.Tensor(sentence2_training_embeddings[i : i + batch_size]).to(device)
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                # Forward pass
                outputs: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average="macro")

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size].to(device)
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size].to(device)

                    # Prepare PH vectors
                    batch_s1_feature_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s1_feature_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)

                    # Forward pass
                    val_outputs: torch.Tensor = self(
                        [
                            s1_embedding,
                            s2_embedding,
                            batch_s1_feature_a,
                            batch_s1_feature_b,
                            batch_s2_feature_a,
                            batch_s2_feature_b,
                        ]
                    )

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels, average="macro")
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.embedding_model.to(device)
        # Clean data
        test_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence1"]
        ]
        test_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_sbert"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_sbert"]), dim=0)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Test cleaning
            test_data[column] = test_data[column].apply(ph_to_tensor)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                output: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
            return final_predictions


# Add PH
class ConvSBERTNegPH(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvSBERTNegPH, self).__init__()
        # Define model layers
        self.num_classes = num_classes
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.conv1 = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(1186, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(65, self.num_classes)  # output layer

    def forward(self, inputs: tuple) -> torch.Tensor:
        x1, x2, num_negation, ph1a, ph1b, ph2a, ph2b = inputs  # ph shape: [x, 260, 3] amd [x, 50, 3]
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
        x_concat = x2 - x1  # [x, 256]
        pha_concat = ph1a - ph2a  # [x, 260, 3]
        phb_concat = ph1b + ph2b  # [x, 50, 3]

        pha_concat_reshaped = pha_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 260*3]
        phb_concat_reshaped = phb_concat.view(x_concat.shape[0], -1)  # Reshape to [x, 50*3]

        # Now you can concatenate x_concat, pha_concat_reshaped, and phb_concat_reshaped along dimension 1
        final_input = torch.cat(
            (x_concat, pha_concat_reshaped, phb_concat_reshaped), dim=1
        )  # [x, 256 + 260*3 + 50*3] ([x, 1186])
        # Feed to forward composition layers
        x = self.fc1(final_input)
        x = self.dropout1(x)
        # Add num_negation to x
        x = torch.cat((x, num_negation), dim=1)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output

    def fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        batch_size: int,
        num_epochs: int,
        device: str,
        verbose: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        self.embedding_model.to(device)
        # Clean data
        training_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence1"]
        ]
        training_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in training_data["sentence2"]
        ]
        validation_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence1"]
        ]
        validation_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in validation_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence1_sbert"]), dim=0)
        sentence2_training_embeddings: torch.Tensor = torch.stack(list(training_data["sentence2_sbert"]), dim=0)
        sentence1_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence1_sbert"]), dim=0)
        sentence2_validation_embeddings: torch.Tensor = torch.stack(list(validation_data["sentence2_sbert"]), dim=0)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            training_data[column] = training_data[column].apply(ph_to_tensor)
            # Validation cleaning
            validation_data[column] = validation_data[column].apply(ph_to_tensor)

        device = torch.device(device)
        self.to(device)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
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
                # Zero the gradients for this batch
                optimizer.zero_grad()
                # Prepare the batch
                s1_embedding: torch.Tensor = torch.Tensor(sentence1_training_embeddings[i : i + batch_size]).to(device)
                s2_embedding: torch.Tensor = torch.Tensor(sentence2_training_embeddings[i : i + batch_size]).to(device)
                # Get the corresponding labels for this batch
                batch_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)

                batch_negations: np.ndarray = (
                    torch.tensor(training_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                    .view(-1, 1)
                    .to(device)
                )

                # Forward pass
                outputs: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_negations,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                # Compute the loss
                loss: float = criterion(outputs, batch_labels)
                # Backpropagation
                optimizer.zero_grad()  # Clear accumulated gradients
                loss.backward()
                # Optimize (update model parameters)
                optimizer.step()
                # Update running loss
                running_loss += loss.item()
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels: np.ndarray = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                # Calculate training accuracy and F1-score
                training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
                training_f1: float = f1_score(all_true_labels, all_predicted_labels, average="macro")

                train_accuracy_values.append(training_accuracy)
                train_f1_values.append(training_f1)

            average_loss: float = running_loss / (len(training_data) / batch_size)
            # Validation
            self.eval()  # Set the model to evaluation mode
            all_val_predicted_labels: list = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    # Prepare the batch for validation
                    s1_embedding: torch.Tensor = sentence1_validation_embeddings[i : i + batch_size].to(device)
                    s2_embedding: torch.Tensor = sentence2_validation_embeddings[i : i + batch_size].to(device)
                    batch_negations: np.ndarray = (
                        torch.tensor(validation_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                        .view(-1, 1)
                        .to(device)
                    )

                    # Prepare PH vectors
                    batch_s1_feature_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s1_feature_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)
                    batch_s2_feature_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                    ).to(device)

                    # Forward pass
                    val_outputs: torch.Tensor = self(
                        [
                            s1_embedding,
                            s2_embedding,
                            batch_negations,
                            batch_s1_feature_a,
                            batch_s1_feature_b,
                            batch_s2_feature_a,
                            batch_s2_feature_b,
                        ]
                    )

                    # Convert validation outputs to class predictions
                    val_class_probabilities = torch.softmax(val_outputs, dim=1)
                    val_predicted_classes = torch.argmax(val_class_probabilities, dim=1)

                    all_val_predicted_labels.extend(val_predicted_classes.cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(validation_data["label"], all_val_predicted_labels)
            val_f1: float = f1_score(validation_data["label"], all_val_predicted_labels, average="macro")
            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        return train_accuracy_values, train_f1_values, val_accuracy_values, val_f1_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.embedding_model.to(device)
        # Clean data
        test_data["sentence1_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence1"]
        ]
        test_data["sentence2_sbert"] = [
            torch.Tensor(self.embedding_model.encode(s.strip())) for s in test_data["sentence2"]
        ]
        # Stack embeddings for batch processing
        sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence1_sbert"]), dim=0)
        sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_data["sentence2_sbert"]), dim=0)
        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Test cleaning
            test_data[column] = test_data[column].apply(ph_to_tensor)

        device = torch.device(device)
        self.to(device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            final_predictions: np.ndarray = np.array([])
            all_true_labels: list = []

            for i in range(0, len(test_data), batch_size):
                # Prepare the batch
                s1_embedding: torch.Tensor = sentence1_testing_embeddings[i : i + batch_size].to(device)
                s2_embedding: torch.Tensor = sentence2_testing_embeddings[i : i + batch_size].to(device)
                batch_negations: np.ndarray = (
                    torch.tensor(test_data["negation"].iloc[i : i + batch_size].values, dtype=torch.long)
                    .view(-1, 1)
                    .to(device)
                )
                true_labels: torch.Tensor = torch.tensor(
                    test_data["label"].iloc[i : i + batch_size].values, dtype=torch.long
                ).to(device)
                # Prepare PH vectors
                batch_s1_feature_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i : i + batch_size], dim=0
                ).to(device)
                output: torch.Tensor = self(
                    [
                        s1_embedding,
                        s2_embedding,
                        batch_negations,
                        batch_s1_feature_a,
                        batch_s1_feature_b,
                        batch_s2_feature_a,
                        batch_s2_feature_b,
                    ]
                )
                class_probabilities = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)

                # Calculate accuracy and F1-score for this batch
                true_labels_cpu: np.ndarray = true_labels.cpu().numpy()
                predicted_classes_cpu: np.ndarray = predicted_classes.cpu().numpy()
                all_true_labels.extend(true_labels_cpu)
                final_predictions = np.append(final_predictions, predicted_classes_cpu)
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

    for name, model_class in [
        ("ConvBBU", ConvBBU(num_classes=3)),
        ("ConvBBUNeg", ConvBBUNeg(num_classes=3)),
        ("ConvBBUPH", ConvBBUPH(num_classes=3)),
        ("ConvBBUNegPH", ConvBBUNegPH(num_classes=3)),
        ("ConvSBERT", ConvSBERT(num_classes=3)),
        ("ConvSBERTNeg", ConvSBERTNeg(num_classes=3)),
        ("ConvSBERTPH", ConvSBERTPH(num_classes=3)),
        ("ConvSBERTNegPH", ConvSBERTNegPH(num_classes=3)),
    ]:
        acc: list = []
        f1: list = []
        precision: list = []
        recall: list = []
        for i in range(30):
            if n is not None and v is not None and t is not None:
                train_df = pd.read_csv("Data/MultiNLI/match_cleaned.csv").head(n)
                valid_df = pd.read_csv("Data/MultiNLI/mismatch_cleaned.csv").head(v)
                test_df = pd.read_csv("Data/contradiction-dataset_cleaned_ph.csv").head(t)
            else:
                train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
                valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
                test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")

            model = model_class

            train_accuracy, train_f1, valid_accuracy, valid_f1 = model.fit(
                train_df, valid_df, BATCH_SIZE, NUM_EPOCHS, DEVICE, verbose=False
            )

            # model_save_path: str = f"Models/{name}.pt"
            # torch.save(model.state_dict(), model_save_path)
            # model.load_state_dict(torch.load(model_save_path))

            predictions = model.predict(test_df, BATCH_SIZE, DEVICE)
            acc.append(accuracy_score(test_df["label"].values, predictions))
            f1.append(f1_score(test_df["label"].values, predictions, average="macro"))
            precision.append(precision_score(test_df["label"].values, predictions, average="macro"))
            recall.append(recall_score(test_df["label"].values, predictions, average="macro"))
        print(f"\t{name}")
        print(
            f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
        )
