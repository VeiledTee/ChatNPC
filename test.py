import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Set a random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

matplotlib.use('TkAgg')


# Load the preprocessed data
data = pd.concat([pd.read_csv('Data/match.csv'), pd.read_csv('Data/mismatch.csv')])
test_data = pd.read_csv('Data/contradiction-dataset.csv')

# Define the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data['sentence1'].iloc[idx]
        hypothesis = self.data['sentence2'].iloc[idx]
        label = 1 if self.data['gold_label'].iloc[idx].lower() == 'contradiction' else 0

        premise_encoded = self.tokenizer(premise, add_special_tokens=True, truncation=True, padding='max_length',
                                         max_length=self.max_length)
        hypothesis_encoded = self.tokenizer(hypothesis, add_special_tokens=True, truncation=True, padding='max_length',
                                            max_length=self.max_length)

        return {
            'sentence1': torch.tensor(premise_encoded['input_ids'], dtype=torch.long),
            'premise_attention_mask': torch.tensor(premise_encoded['attention_mask'], dtype=torch.long),
            'sentence2': torch.tensor(hypothesis_encoded['input_ids'], dtype=torch.long),
            'sentence2_attention_mask': torch.tensor(hypothesis_encoded['attention_mask'], dtype=torch.long),
            'gold_label': torch.tensor(label, dtype=torch.float)
        }


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.dropout(self.embedding(input_ids))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, torch.sum(attention_mask, dim=1),
                                                            batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


# Create the custom dataset
dataset = CustomDataset(data, tokenizer, max_length=64)

# Define batch size and create DataLoader
BATCH_SIZE = 32
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE)

# hyperparameters
vocab_size = tokenizer.vocab_size
embedding_dim = 100
hidden_dim = 256
output_dim = 1
dropout = 0.5

# Initialize the model and move it to the GPU if available
model = BiLSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, dropout)
device = torch.device('cpu')
model = model.to(device)

print("Model on CPU")

# Define the loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())


# Function to calculate accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# Function to train the model
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        text, text_attention_mask = batch['sentence1'], batch['premise_attention_mask']
        optimizer.zero_grad()
        predictions = model(text, text_attention_mask).squeeze(1)
        loss = criterion(predictions, batch['gold_label'].to(device))
        acc = binary_accuracy(predictions, batch['gold_label'].to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Function to evaluate the model
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in iterator:
            text, text_attention_mask = batch['sentence1'], batch['premise_attention_mask']
            predictions = model(text, text_attention_mask).squeeze(1)
            loss = criterion(predictions, batch['gold_label'].to(device))
            acc = binary_accuracy(predictions, batch['gold_label'].to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def model_test(model, iterator, criterion):
    model.eval()  # Set the model to evaluation mode

    test_loss = 0.0
    correct_preds = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['sentence1']
            attention_mask = batch['premise_attention_mask']
            labels = batch['gold_label']

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels.float())
            test_loss += loss.item()

            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct_preds += (rounded_preds == labels).sum().item()

            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(iterator)
    test_accuracy = correct_preds / (len(iterator.dataset) * 1.0)
    test_f1 = f1_score(all_labels, all_preds)

    return test_loss, test_accuracy, test_f1


def calculate_f1(predictions, y):
    rounded_preds = torch.round(torch.sigmoid(predictions))
    return f1_score(y.cpu().numpy(), rounded_preds.cpu().numpy())


# Train the model
N_EPOCHS = 2
train_losses = []
train_accuracies = []
train_f1_scores = []
valid_losses = []
valid_accuracies = []
valid_f1_scores = []

for epoch in range(N_EPOCHS):
    print(f'Epoch: {epoch + 1:02}')
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    with torch.no_grad():
        train_preds = torch.cat(
            [model(batch['sentence1'], batch['premise_attention_mask']) for batch in train_iterator])
    train_f1 = calculate_f1(train_preds, torch.cat([batch['gold_label'] for batch in train_iterator]))
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F1: {train_f1:.3f}')

    valid_loss, valid_acc = evaluate(model, val_iterator, criterion)
    with torch.no_grad():
        valid_preds = torch.cat([model(batch['sentence1'], batch['premise_attention_mask']) for batch in val_iterator])
    valid_f1 = calculate_f1(valid_preds, torch.cat([batch['gold_label'] for batch in val_iterator]))
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Val. F1: {valid_f1:.3f}')

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    train_f1_scores.append(train_f1)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)
    valid_f1_scores.append(valid_f1)

test_loss, test_accuracy, test_f1 = model_test(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_accuracy * 100:.2f}% | Test F1: {test_f1:.3f}')

plt.figure(figsize=(12, 8))

# Plot training and validation accuracy
plt.subplot(3, 1, 1)
plt.plot(range(1, N_EPOCHS+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, N_EPOCHS+1), valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(3, 1, 2)
plt.plot(range(1, N_EPOCHS+1), train_losses, label='Training Loss')
plt.plot(range(1, N_EPOCHS+1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation F1 score
plt.subplot(3, 1, 3)
plt.plot(range(1, N_EPOCHS+1), train_f1_scores, label='Training F1 Score')
plt.plot(range(1, N_EPOCHS+1), valid_f1_scores, label='Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Training and Validation F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
