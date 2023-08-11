from typing import Dict, Tuple

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np

from variables import DEVICE
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


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


# # Set a random seed for reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
print()


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
    input_ids = torch.nn.functional.pad(input_ids, (0, 64 - input_ids.size(1)))
    attention_mask = torch.nn.functional.pad(attention_mask, (0, 64 - attention_mask.size(1)))

    # Obtain the BERT embeddings
    with torch.no_grad():
        bert_outputs: Tuple[torch.Tensor] = bert_model(input_ids, attention_mask=attention_mask)
        embeddings: torch.Tensor = bert_outputs.last_hidden_state  # Extract the last hidden state

    return embeddings


class SentenceClassifier(nn.Module):
    def __init__(self, input_dim=786, output_dim=1):
        super(SentenceClassifier, self).__init__()

        # Add a convolutional layer
        self.conv_layer = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.maxpool_layer = nn.MaxPool1d(kernel_size=2)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.output_layer = nn.Linear(32, output_dim)

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = x.permute(0, 2, 1)  # Permute to (batch_size, channels, sequence_length)
        x = self.conv_layer(x)
        x = self.maxpool_layer(x)

        # Apply tanh activation
        x = self.tanh(x)

        # Apply global average pooling to collapse
        x = torch.mean(x, dim=-1)

        x = self.output_layer(x)

        # Apply sigmoid activation
        x = self.sigmoid(x)

        return x


n = 30
t = 6
num_epochs = 10
batch_size = 256

train_df = pd.read_csv("Data/match.csv").head(n)
test_df = pd.read_csv("Data/match.csv").head(t)
train_df["label"] = np.where(train_df["gold_label"] == "contradiction", 1, 0)  # label cleaning
test_df["label"] = np.where(test_df["gold_label"] == "contradiction", 1, 0)  # label cleaning

train_df["embeddings"] = train_df.apply(apply_get_bert_embeddings, axis=1)
test_df["embeddings"] = test_df.apply(apply_get_bert_embeddings, axis=1)

# Sample BERT embeddings (replace this with the actual embeddings from get_bert_embeddings function)
train_bert_embeddings = torch.stack(list(train_df["embeddings"]), dim=0)
test_bert_embeddings = torch.stack(list(test_df["embeddings"]), dim=0)
# Initialize the model
model = SentenceClassifier()
device = torch.device(DEVICE)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss().to(device)

print(f"Model on {device}")

# Define the optimizer (e.g., Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    all_predicted_labels = []
    all_true_labels = []

    for i in range(0, len(train_df), batch_size):
        # Prepare the batch
        batch_embeddings = train_bert_embeddings[i: i + batch_size]

        # Get the corresponding labels for this batch
        batch_labels = train_df["label"].iloc[i: i + batch_size].values
        batch_labels = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1)

        # Move tensors to the device
        batch_embeddings = batch_embeddings.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = model(batch_embeddings)

        # Compute the loss
        loss = criterion(outputs, batch_labels)

        # Backpropagation
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

    # print(all_true_labels)
    # print(all_predicted_labels)
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)

    # Print training metrics for this epoch
    average_loss = running_loss / (len(train_df) / batch_size)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    test_bert_embeddings = test_bert_embeddings.to(device)
    output = model(test_bert_embeddings)
    print(output)
    predicted_labels = (output >= 0.2).float().cpu().numpy()
    true_labels = test_df["label"].values

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
