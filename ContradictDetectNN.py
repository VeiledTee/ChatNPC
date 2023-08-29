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


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data["sentence1"].iloc[idx]
        hypothesis = self.data["sentence2"].iloc[idx]
        label = 1 if self.data["gold_label"].iloc[idx].lower() == "contradiction" else 0

        premise_encoded = self.tokenizer(
            premise, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length
        )
        hypothesis_encoded = self.tokenizer(
            hypothesis, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length
        )

        return {
            "sentence1": torch.tensor(premise_encoded["input_ids"], dtype=torch.long),
            "premise_attention_mask": torch.tensor(premise_encoded["attention_mask"], dtype=torch.long),
            "sentence2": torch.tensor(hypothesis_encoded["input_ids"], dtype=torch.long),
            "sentence2_attention_mask": torch.tensor(hypothesis_encoded["attention_mask"], dtype=torch.long),
            "gold_label": torch.tensor(label, dtype=torch.float),
        }


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, num_layers):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(embedding_dim if i == 0 else hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
                for i in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        print(
            "Dimensionality of word embeddings:", self.embedding.embedding_dim
        )  # Print the dimensionality of word embeddings
        embedded = self.dropout(self.embedding(input_ids))

        lstm_input = embedded
        for lstm_layer in self.lstm_layers:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                lstm_input, torch.sum(attention_mask, dim=1), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = lstm_layer(packed_embedded)
            lstm_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            lstm_input = lstm_output

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


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


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        text, text_attention_mask = batch["sentence1"], batch["premise_attention_mask"]
        optimizer.zero_grad()
        predictions = model(text, text_attention_mask).squeeze(1)
        loss = criterion(predictions, batch["gold_label"].to(device))
        acc = binary_accuracy(predictions, batch["gold_label"].to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in iterator:
            text, text_attention_mask = batch["sentence1"], batch["premise_attention_mask"]
            predictions = model(text, text_attention_mask).squeeze(1)
            loss = criterion(predictions, batch["gold_label"].to(device))
            acc = binary_accuracy(predictions, batch["gold_label"].to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def model_test(model, iterator, criterion):
    model.eval()  # Set the model to evaluation mode

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["sentence1"]
            attention_mask = batch["premise_attention_mask"]
            labels = batch["gold_label"]

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions.squeeze(), labels.float().squeeze())
            test_loss += loss.item()

            rounded_preds = torch.round(torch.sigmoid(predictions))
            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(iterator)
    test_accuracy = binary_accuracy(torch.tensor(all_preds), torch.tensor(all_labels))

    test_f1 = f1_score(all_labels, all_preds)

    return test_loss, test_accuracy, test_f1


def calculate_f1(predictions, true):
    # Round the predictions to obtain binary values (0 or 1)
    print(predictions)
    rounded_preds = torch.round(torch.sigmoid(predictions)).cpu().numpy()
    y_true = true.cpu().numpy()
    print(f"Length Predictions: {len(y_true)}")
    print(f"Predicted Sum:      {rounded_preds.sum()}")
    print(f"True Sum:           {sum(y_true)}")

    # Calculate true positives, false positives, and false negatives
    tp = ((rounded_preds == 1) & (y_true == 1)).sum()
    fp = ((rounded_preds == 1) & (y_true == 0)).sum()
    fn = ((rounded_preds == 0) & (y_true == 1)).sum()

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1


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
    model = BertModel.from_pretrained(model_name)

    # Tokenize the input sentence
    tokens = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")

    # Get the model output
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the representation of [CLS] token (sentence embedding)
    sentence_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
    return sentence_embedding.squeeze()


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


def plot_training_history(train_accuracy_list: list, val_accuracy_list: list, train_f1_list: list, val_f1_list: list, number_of_epochs: int):
    # Create x-axis values (epochs)
    x_axis: list = list(range(1, number_of_epochs + 1))

    # Create separate plots for accuracy and F1-score
    plt.figure(figsize=(12, 4))

    # Training accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, train_accuracy_list, label='Training Accuracy', marker='o')
    plt.plot(x_axis, val_accuracy_list, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Training F1-score plot
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, train_f1_list, label='Training F1 Score', marker='o')
    plt.plot(x_axis, val_f1_list, label='Validation F1 Score', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


class SentenceClassifier(nn.Module):
    def __init__(self):
        super(SentenceClassifier, self).__init__()

        # Define your layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(256, 64)  # composition layer after concatenation
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer 1
        self.fc2 = nn.Linear(64, 1)  # output layer

    def forward(self, inputs):
        x1, x2, num_negation = inputs

        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)

        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate the representations of x1 and x2
        # concatenated = torch.cat((x1, x2), dim=1)
        concatenated = x2 - x1
        x = self.fc1(concatenated)
        x = self.dropout1(x)
        final_layer_output = torch.sigmoid(self.fc2(x))

        return final_layer_output


class SentenceSiameseClassifier(nn.Module):
    def __init__(self):
        super(SentenceSiameseClassifier, self).__init__()

        # Define your layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(2, 1)  # composition layer

    def forward(self, inputs):
        x1, x2, num_negation = inputs

        # Apply convolution and max pooling
        x1 = F.relu(self.conv1(x1.unsqueeze(2)))  # Add a channel dimension
        x1 = self.maxpool(x1)
        x2 = F.relu(self.conv1(x2.unsqueeze(2)))  # Add a channel dimension
        x2 = self.maxpool(x2)

        # Apply tanh activation function to x1 and x2 tensors
        x1 = torch.tanh(x1)
        x2 = torch.tanh(x2)

        # Flatten the convolutional outputs
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Calculate the Euclidean distance between embeddings
        distance = torch.norm(x1 - x2, dim=1, keepdim=True)

        # Concatenate the distance with the additional feature
        concatenated = torch.cat((distance, num_negation), dim=1)

        # Pass through the composition layer
        x = self.fc1(concatenated)
        final_layer_output = torch.sigmoid(x)

        return final_layer_output


# matplotlib.use('TkAgg')
#
# # Load the preprocessed data
# # data = pd.concat([pd.read_csv('Data/match.csv'), pd.read_csv('Data/mismatch.csv')])
# # data = pd.read_csv('Data/match.csv')
# data = pd.read_csv('Data/contradiction-dataset.csv')
# test_data = pd.read_csv('Data/contradiction-dataset.csv')
#
# # Define the BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
# # Create the custom dataset
# dataset = CustomDataset(data, tokenizer, max_length=128)
# test_dataset = CustomDataset(test_data, tokenizer, max_length=128)
#
# # Define batch size and create DataLoader
# BATCH_SIZE = 32
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
# train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
# # hyperparameters
# vocab_size = tokenizer.vocab_size
# embedding_dim = 512
# hidden_dim = 256
# output_dim = 1
# dropout = 0
# num_layers = 128
#
# # Initialize the model and move it to the GPU if available
# model = BiLSTMModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, num_layers=num_layers)
# device = torch.device('cpu')
# model = model.to(device)
#
# print("Model on CPU")
#
# # Define the loss and optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters())
#
# # Train the model
# N_EPOCHS = 10
# train_losses = []
# train_accuracies = []
# train_f1_scores = []
# valid_losses = []
# valid_accuracies = []
# valid_f1_scores = []
#
# for epoch in range(N_EPOCHS):
#     train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
#     with torch.no_grad():
#         train_preds = torch.cat(
#             [model(batch['sentence1'], batch['premise_attention_mask']) for batch in train_iterator])
#     print(f'Epoch: {epoch + 1:02}')
#     train_f1 = calculate_f1(train_preds, torch.cat([batch['gold_label'] for batch in train_iterator]))
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F1: {train_f1:.3f}')
#
#     valid_loss, valid_acc = evaluate(model, val_iterator, criterion)
#     with torch.no_grad():
#         valid_preds = torch.cat([model(batch['sentence1'], batch['premise_attention_mask']) for batch in val_iterator])
#     valid_f1 = calculate_f1(valid_preds, torch.cat([batch['gold_label'] for batch in val_iterator]))
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Val. F1: {valid_f1:.3f}')
#
#     train_losses.append(train_loss)
#     train_accuracies.append(train_acc)
#     train_f1_scores.append(train_f1)
#     valid_losses.append(valid_loss)
#     valid_accuracies.append(valid_acc)
#     valid_f1_scores.append(valid_f1)
#
# test_loss, test_accuracy = evaluate(model, test_iterator, criterion)
# with torch.no_grad():
#     test_preds = torch.cat([model(batch['sentence1'], batch['premise_attention_mask']) for batch in test_iterator])
# test_f1 = calculate_f1(test_preds, torch.cat([batch['gold_label'] for batch in test_iterator]))
# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_accuracy * 100:.2f}% | Test F1: {test_f1:.3f}')
#
# plt.figure(figsize=(12, 8))
#
# # Plot training and validation accuracy
# plt.subplot(3, 1, 1)
# plt.plot(range(1, N_EPOCHS+1), train_accuracies, label='Training Accuracy')
# plt.plot(range(1, N_EPOCHS+1), valid_accuracies, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
#
# # Plot training and validation loss
# plt.subplot(3, 1, 2)
# plt.plot(range(1, N_EPOCHS+1), train_losses, label='Training Loss')
# plt.plot(range(1, N_EPOCHS+1), valid_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
#
# # Plot training and validation F1 score
# plt.subplot(3, 1, 3)
# plt.plot(range(1, N_EPOCHS+1), train_f1_scores, label='Training F1 Score')
# plt.plot(range(1, N_EPOCHS+1), valid_f1_scores, label='Validation F1 Score')
# plt.xlabel('Epoch')
# plt.ylabel('F1 Score')
# plt.title('Training and Validation F1 Score')
# plt.legend()
#
# plt.tight_layout()
# plt.show()


if __name__ == "__main__":
    num_epochs = 10
    batch_size = 64

    # Load and preprocess the data
    # n = 800
    # v = 100
    # t = 100
    n = None
    v = None
    t = None
    if n is not None and v is not None and t is not None:
        train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned.csv.csv").head(n)
        valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned.csv").head(v)
        test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned.csv").head(t)
    else:
        # train_df = pd.concat([
        #     # pd.read_csv("Data/match_cleaned.csv"),
        #     pd.read_csv("Data/mismatch_cleaned.csv"),
        #     # pd.read_csv("Data/SemEval2014T1/train_cleaned.csv"),
        #     # pd.read_csv("Data/SemEval2014T1/valid_cleaned.csv"),
        # ], ignore_index=True)
        # valid_df = pd.read_csv("Data/contradiction-dataset_cleaned_ph.csv")
        # test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned.csv")
        train_df = pd.read_csv("Data/match_cleaned.csv")
        valid_df = pd.read_csv("Data/mismatch_cleaned.csv")
        test_df = pd.read_csv("Data/contradiction-dataset_cleaned_ph.csv")

    # correctly format tensors
    train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(str_to_tensor)
    train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(str_to_tensor)
    valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(str_to_tensor)
    valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(str_to_tensor)
    test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(str_to_tensor)
    test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(str_to_tensor)
    # stack tensors to pass to model
    train_bert_embeddings_sentence1 = torch.stack(list(train_df["sentence1_embeddings"]), dim=0)
    train_bert_embeddings_sentence2 = torch.stack(list(train_df["sentence2_embeddings"]), dim=0)
    valid_bert_embeddings_sentence1 = torch.stack(list(valid_df["sentence1_embeddings"]), dim=0)
    valid_bert_embeddings_sentence2 = torch.stack(list(valid_df["sentence2_embeddings"]), dim=0)
    test_bert_embeddings_sentence1 = torch.stack(list(test_df["sentence1_embeddings"]), dim=0)
    test_bert_embeddings_sentence2 = torch.stack(list(test_df["sentence2_embeddings"]), dim=0)

    # Initialize empty lists to store training and validation metrics
    train_accuracy_values = []
    train_f1_values = []
    val_accuracy_values = []
    val_f1_values = []
    tests_acc: list = []
    tests_f1: list = []

    # for _ in range(2500):  # stat significance testing
    # load model
    model = SentenceClassifier()
    device = torch.device(DEVICE)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Model on {device}")

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        all_predicted_labels = []
        all_true_labels = []

        for i in range(0, len(train_df), batch_size):
            # Prepare the batch
            s1_embedding = train_bert_embeddings_sentence1[i : i + batch_size]
            s2_embedding = train_bert_embeddings_sentence2[i : i + batch_size]
            # Get the corresponding labels for this batch
            batch_labels = train_df["label"].iloc[i : i + batch_size].values
            batch_labels = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1)
            # Get additional feature values
            num_negations = train_df["negation"].iloc[i : i + batch_size].values
            batch_negations = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1)

            # s1_ph_features = persistent_homology_features(list(train_df["sentence1"].iloc[i : i + batch_size]))
            # dim_0_s1_features = [item[0] for item in s1_ph_features]
            # # dim_1_s1_features = [item[1] for item in s1_ph_features]
            # batch_s1_feature_a = torch.tensor(np.array(dim_0_s1_features)).to(device)
            # # batch_s1_feature_b = torch.tensor(np.array(dim_1_s1_features)).to(device)
            # print(batch_s1_feature_a.shape)
            # print(batch_s1_feature_b.shape)

            # s2_ph_features = persistent_homology_features(list(train_df["sentence2"].iloc[i : i + batch_size]))
            # dim_0_s2_features = [item[0] for item in s2_ph_features]
            # # dim_1_s2_features = [item[1] for item in s2_ph_features]
            # batch_s2_feature_a = torch.tensor(np.array(dim_0_s2_features)).to(device)
            # # batch_s2_feature_b = torch.tensor(np.array(dim_1_s2_features)).to(device)
            # print(batch_s2_feature_a.shape)
            # print(batch_s2_feature_b.shape)

            # Move tensors to the device
            s1_embedding = s1_embedding.to(device)
            s2_embedding = s2_embedding.to(device)
            batch_labels = batch_labels.to(device)
            batch_negations = batch_negations.to(device)
            # Forward pass
            # outputs = model([s1_embedding, s2_embedding, batch_negations, batch_s1_feature_a, batch_s1_feature_b, batch_s2_feature_a, batch_s2_feature_b])
            outputs = model([s1_embedding, s2_embedding, batch_negations])
            # Compute the loss
            loss = criterion(outputs, batch_labels)
            # Backpropagation
            optimizer.zero_grad()  # Clear accumulated gradients
            loss.backward()
            # Optimize (update model parameters)
            optimizer.step()
            # Update running loss
            running_loss += loss.item()
            # Convert outputs to binary predictions (0 or 1)
            predicted_labels = (outputs >= 0.5).float().view(-1).cpu().numpy()
            true_labels = batch_labels.view(-1).cpu().numpy()
            all_predicted_labels.extend(predicted_labels)
            all_true_labels.extend(true_labels)
        # Calculate training accuracy and F1-score
        train_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        train_f1 = f1_score(all_true_labels, all_predicted_labels)

        # Print training metrics for this epoch
        average_loss = running_loss / (len(train_df) / batch_size)

        # Validation
        model.eval()  # Set the model to evaluation mode
        all_val_predicted_labels = []

        with torch.no_grad():
            for i in range(0, len(valid_df), batch_size):
                # Prepare the batch for validation
                s1_embedding = valid_bert_embeddings_sentence1[i : i + batch_size]
                s2_embedding = valid_bert_embeddings_sentence2[i : i + batch_size]
                batch_negations = valid_df["negation"].iloc[i : i + batch_size].values

                # Move tensors to the device
                s1_embedding = s1_embedding.to(device)
                s2_embedding = s2_embedding.to(device)
                batch_negations = (
                    torch.tensor(batch_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
                )

                # Forward pass for validation
                val_outputs = model([s1_embedding, s2_embedding, batch_negations])

                # Convert validation outputs to binary predictions (0 or 1)
                val_predicted_labels = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
                all_val_predicted_labels.extend(val_predicted_labels)

        # Calculate validation accuracy and F1-score
        val_accuracy = accuracy_score(valid_df["label"], all_val_predicted_labels)
        val_f1 = f1_score(valid_df["label"], all_val_predicted_labels)
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"\tTraining   | Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, Loss: {average_loss:.4f}")
        print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        train_accuracy_values.append(train_accuracy)
        train_f1_values.append(train_f1)
        val_accuracy_values.append(val_accuracy)
        val_f1_values.append(val_f1)

    plot_training_history(train_accuracy_values, val_accuracy_values, train_f1_values, val_f1_values, num_epochs)

    # model_save_path: str = "Models/test1.pt"
    # torch.save(model.state_dict(), model_save_path)
    # model.load_state_dict(torch.load(model_save_path))

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = np.array([])
        for i in range(0, len(test_df), batch_size):
            # Prepare the batch
            s1_embedding = test_bert_embeddings_sentence1[i : i + batch_size].to(device)
            s2_embedding = test_bert_embeddings_sentence2[i : i + batch_size].to(device)
            # Get additional feature values
            num_negations = test_df["negation"].iloc[i : i + batch_size].values
            batch_negations = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1).to(device)
            output = model([s1_embedding, s2_embedding, batch_negations])
            predicted_labels = (output >= 0.5).float().cpu().numpy()
            predictions = np.append(predictions, predicted_labels)

        true_labels = test_df["label"].values
        train_accuracy = accuracy_score(true_labels, predictions)
        train_f1 = f1_score(true_labels, predictions)
        tests_acc.append(train_accuracy)
        tests_f1.append(train_f1)
        print(f"Test Accuracy: {train_accuracy:.4f}, Test F1 Score: {train_f1:.4f}")
        print(f"Percent of positive class: {sum(true_labels) / len(true_labels):.4f}%")


    # print(f"Avg acc over 30 runs: {sum(tests_acc) / len(tests_acc)}")
    # print(f"Best Accuracy:        {max(tests_acc)}")
    # print(f"Worst Accuracy:       {min(tests_acc)}")
    # print(f"Avg f1 over 30 runs:  {sum(tests_f1) / len(tests_f1)}")
    # print(f"Best F1 Score:        {max(tests_f1)}")
    # print(f"Worst F1 Score:       {min(tests_f1)}")
    """
    Avg acc over 30 runs: 0.7149188514357054
    Best Accuracy:        0.7640449438202247
    Worst Accuracy:       0.649812734082397
    Avg f1 over 30 runs:  0.8275341820334495
    Best F1 Score:        0.8630434782608696
    Worst F1 Score:       0.7823050058207216
    """
    # print(f"=====\n\tAccuracy Breakdown\n=====")
    # analyze_float_list(tests_acc)
    # print(f"=====\n\tF1 Score Breakdown\n=====")
    # analyze_float_list(tests_f1)
    """
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
    # num_epochs = 10
    # batch_size = 64
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
    # criterion = nn.BCEWithLogitsLoss().to(device)
    # optimizer = optim.Adam(siamese_model.parameters(), lr=0.1)
    #
    # print(f"Model on {device}")
    #
    # for epoch in range(num_epochs):
    #     siamese_model.train()  # Set the model to training mode
    #     running_loss = 0.0
    #     all_true_labels = []
    #     all_predicted_labels = []
    #
    #     for i in range(0, len(train_df), batch_size):
    #         # Prepare the batch
    #         s1_embedding = train_bert_embeddings_sentence1[i: i + batch_size].to(device)
    #         s2_embedding = train_bert_embeddings_sentence2[i: i + batch_size].to(device)
    #         # Get the corresponding labels for this batch
    #         batch_labels = train_df["label"].iloc[i: i + batch_size].values
    #         batch_labels = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1).to(device)
    #         # Get additional feature values
    #         num_negations = train_df["negation"].iloc[i: i + batch_size].values
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
    #         predicted_labels = (outputs >= 0.5).float().view(-1).cpu().numpy()
    #         true_labels = batch_labels.view(-1).cpu().numpy()
    #         all_true_labels.extend(true_labels)
    #         all_predicted_labels.extend(predicted_labels)
    #
    #     # Calculate training accuracy and F1-score
    #     accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    #     f1 = f1_score(all_true_labels, all_predicted_labels)
    #
    #     # Print training metrics for this epoch
    #     average_loss = running_loss / (len(train_df) / batch_size)
    #     print(
    #         f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}"
    #     )
    #
    #     # Validation
    #     siamese_model.eval()  # Set the model to evaluation mode
    #     all_val_predicted_labels = []
    #
    #     with torch.no_grad():
    #         for i in range(0, len(valid_df), batch_size):
    #             # Prepare the batch for validation
    #             s1_embedding = valid_bert_embeddings_sentence1[i: i + batch_size].to(device)
    #             s2_embedding = valid_bert_embeddings_sentence2[i: i + batch_size].to(device)
    #             batch_negations = (
    #                 torch.tensor(valid_df["negation"].iloc[i: i + batch_size].values, dtype=torch.float32)
    #                 .view(-1, 1)
    #                 .to(device)
    #             )
    #
    #             # Forward pass for validation
    #             val_outputs = siamese_model([s1_embedding, s2_embedding, batch_negations])
    #
    #             # Convert validation outputs to binary predictions (0 or 1)
    #             val_predicted_labels = (val_outputs >= 0.5).float().view(-1).cpu().numpy()
    #             all_val_predicted_labels.extend(val_predicted_labels)
    #
    #     # Calculate validation accuracy and F1-score
    #     val_accuracy = accuracy_score(valid_df["label"], all_val_predicted_labels)
    #     val_f1 = f1_score(valid_df["label"], all_val_predicted_labels)
    #
    #     print(f"\tValidation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}")
    #
    # # Testing
    # siamese_model.eval()  # Set the model to evaluation mode
    # predictions = np.array([])
    #
    # with torch.no_grad():
    #     for i in range(0, len(test_df), batch_size):
    #         # Prepare the batch
    #         s1_embedding = test_bert_embeddings_sentence1[i: i + batch_size].to(device)
    #         s2_embedding = test_bert_embeddings_sentence2[i: i + batch_size].to(device)
    #         batch_negations = (
    #             torch.tensor(test_df["negation"].iloc[i: i + batch_size].values, dtype=torch.float32)
    #             .view(-1, 1)
    #             .to(device)
    #         )
    #         output = siamese_model([s1_embedding, s2_embedding, batch_negations])
    #         predicted_labels = (output >= 0.5).float().cpu().numpy()
    #         predictions = np.append(predictions, predicted_labels)
    #
    # true_labels = test_df["label"].values
    # accuracy = accuracy_score(true_labels, predictions)
    # f1 = f1_score(true_labels, predictions)
    #
    # print(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")
