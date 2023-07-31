import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_output, attention_mask):
        # Compute attention scores
        attention_scores = torch.bmm(lstm_output.float(), attention_mask.unsqueeze(2)).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Calculate the weighted sum using attention weights
        attention_output = torch.bmm(lstm_output.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)

        return attention_output

    def forward(self, text, text_attention_mask):
        embedded = self.dropout(self.embedding(text))
        lstm_output, _ = self.lstm(embedded)

        # Apply attention mechanism
        attention_output = self.attention(lstm_output, text_attention_mask)

        return self.fc(attention_output)
