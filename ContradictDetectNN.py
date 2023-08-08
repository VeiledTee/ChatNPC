import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from convert_dataset import load_txt_file_to_dataframe


class SiameseNetworkWithBERT(nn.Module):
    def __init__(self, hidden_size):
        super(SiameseNetworkWithBERT, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.encoder = nn.LSTM(self.bert_model.config.hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        premise_outputs = self.bert_model(input_ids=input_ids["premise"], attention_mask=attention_mask["premise"])
        premise_pooled_output = premise_outputs.pooler_output  # Extract the pooled output

        hypothesis_outputs = self.bert_model(input_ids=input_ids["hypothesis"], attention_mask=attention_mask["hypothesis"])
        hypothesis_pooled_output = hypothesis_outputs.pooler_output  # Extract the pooled output

        # Concatenate the premise and hypothesis embeddings along the batch dimension
        concatenated_embeds = torch.cat((premise_pooled_output, hypothesis_pooled_output), dim=0)

        print(concatenated_embeds.shape)
        return concatenated_embeds


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        sentence1 = self.dataframe.iloc[index]["sentence1"]
        sentence2 = self.dataframe.iloc[index]["sentence2"]
        label = 1 if self.dataframe.iloc[index]["gold_label"].lower() == 'contradiction' else 0

        # Tokenize the sentences using the BERT tokenizer
        encoded_sentences = self.tokenizer(
            sentence1,
            sentence2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoded_sentences['input_ids'].squeeze()
        attention_mask = encoded_sentences['attention_mask'].squeeze()

        return {'premise': input_ids, 'hypothesis': input_ids}, {'premise': attention_mask, 'hypothesis': attention_mask}, torch.tensor(label, dtype=torch.float)


model = SiameseNetworkWithBERT(hidden_size=128)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set maximum length for tokenized inputs (adjust as needed based on your model)
max_length = 128

dataset_descriptors: list = ['match', 'mismatch']
dataframes: list = []

for descriptor in dataset_descriptors:
    df = load_txt_file_to_dataframe(descriptor)
    dataframes.append(df)

# Concatenate all the dataframes into a final dataframe
df = pd.concat(dataframes, ignore_index=True)

# Create the custom dataset
train_dataset = CreateDataset(df, tokenizer, max_length)
val_dataset = CreateDataset(pd.read_csv('Data/contradiction-dataset.csv'), tokenizer, max_length)

# Set batch size for training
batch_size = 32

# Step 4: Create DataLoader for training data
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Step 4: Create DataLoader for validation data
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch: {epoch}")
    for premise, hypothesis, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(premise, hypothesis)
        print(outputs.shape)
        print(outputs.squeeze().shape)
        print(labels.shape)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for premise, hypothesis, labels in val_loader:
            outputs = model(premise, hypothesis)
            predicted = (outputs >= 0.5).long()
            total_correct += (predicted.squeeze() == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {accuracy:.4f}")
