import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ContradictDetectNN import embedding_to_tensor
from variables import DEVICE
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import torch
import numpy as np
from torch import Tensor
from torch.nn import Module
from typing import List
from numpy import ndarray
import torch.optim as optim


class SiameseContrastiveModel(nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super(SiameseContrastiveModel, self).__init__()
        self.embedding_dim = embedding_dim
        # Siamese network consists of two identical subnetworks
        self.subnetwork = nn.Sequential(
            nn.Linear(embedding_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
        )

    def forward_one(self, x):
        return self.subnetwork(x)

    def forward(self, input1, input2):
        # Forward pass for the first sentence
        output1 = self.forward_one(input1)

        # Forward pass for the second sentence (shared weights)
        output2 = self.forward_one(input2)

        return output1, output2

    def fit(self, training_data: pd.DataFrame, batch_size: int, num_epochs: int, device: str, margin: float,
            verbose: bool = False) -> list[float]:
        training_data["sentence1_embeddings"] = training_data["sentence1_embeddings"].apply(embedding_to_tensor)
        training_data["sentence2_embeddings"] = training_data["sentence2_embeddings"].apply(embedding_to_tensor)

        sentence1_training_embeddings = torch.stack(list(training_data["sentence1_embeddings"]), dim=0)
        sentence2_training_embeddings = torch.stack(list(training_data["sentence2_embeddings"]), dim=0)

        device = torch.device(device)
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        train_loss_values = []

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss = 0.0

            for i in range(0, len(training_data), batch_size):
                optimizer.zero_grad()  # Zero the gradients for this batch

                s1_embedding = sentence1_training_embeddings[i: i + batch_size].to(device)
                s2_embedding = sentence2_training_embeddings[i: i + batch_size].to(device)

                output1, output2 = self(s1_embedding, s2_embedding)

                # Calculate similarity scores using cosine similarity
                similarity_scores = F.cosine_similarity(output1, output2)
                target_labels: torch.Tensor = torch.tensor(
                    training_data["label"].iloc[i: i + batch_size].values, dtype=torch.long
                ).to(device)

                # Calculate the contrastive loss
                loss = torch.mean((1 - target_labels) * similarity_scores ** 2 + target_labels * torch.clamp(
                    margin - similarity_scores, min=0.0) ** 2)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            average_loss = running_loss / (len(training_data) / batch_size)

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

            train_loss_values.append(average_loss)

        return train_loss_values

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str, similarity_threshold: float = 0.5) -> List[
        int]:
        # Clean data
        test_data["sentence1_embeddings"] = test_data["sentence1_embeddings"].apply(embedding_to_tensor)
        test_data["sentence2_embeddings"] = test_data["sentence2_embeddings"].apply(embedding_to_tensor)

        device = torch.device(device)
        self.to(device)

        predicted_classes: List[int] = []

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode

            for i in range(len(test_data)):
                # Extract embeddings for individual records
                s1_embedding = test_data["sentence1_embeddings"].iloc[i].unsqueeze(0)
                s2_embedding = test_data["sentence2_embeddings"].iloc[i].unsqueeze(0)

                # Move tensors to the device
                s1_embedding = s1_embedding.to(device)
                s2_embedding = s2_embedding.to(device)

                # Forward pass for validation
                output1, output2 = self(s1_embedding, s2_embedding)
                # print(output1.shape, output2.shape)
                # Calculate similarity score using cosine similarity
                similarity_score = cosine_similarity(output1.cpu().numpy(), output2.cpu().numpy())[0][0]

                # Apply threshold and convert to 0 or 1 for each individual record
                prediction = 1 if similarity_score >= similarity_threshold else 0
                # print(prediction, similarity_score, test_data['label'].values[i])
                predicted_classes.append(prediction)

        return predicted_classes


"""
train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")

train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(embedding_to_tensor)
train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(embedding_to_tensor)
valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(embedding_to_tensor)
valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(embedding_to_tensor)
test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(embedding_to_tensor)
test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(embedding_to_tensor)

# # Split the DataFrame based on the 'Label' column
# label_values = train_df['label'].unique()  # [0 1 2]
# dfs = {}
# for label_value in label_values:
#     dfs[label_value] = train_df[train_df['label'] == label_value]
#
# neutral = dfs[0]
# entailment = dfs[1]
# contradiction = dfs[2]
#
# for df in [neutral, entailment, contradiction]:
#     min_similarity = float('inf')
#     max_similarity = float('-inf')
#     similarity_sum = 0.0
#     for t1, t2 in zip(df['sentence1_embeddings'], df['sentence2_embeddings']):
#         t1 = t1.numpy()
#         t2 = t2.numpy()
#
#         # Reshape the tensors if needed (e.g., for 1D tensors)
#         if len(t1.shape) == 1:
#             t1 = t1.reshape(1, -1)
#         if len(t2.shape) == 1:
#             t2 = t2.reshape(1, -1)
#
#         # Calculate cosine similarity using sklearn's cosine_similarity function
#         similarity_matrix = cosine_similarity(t1, t2)
#
#         # Get the cosine similarity value (it's a matrix, but we only have one value)
#         similarity = similarity_matrix[0, 0]
#
#         # Update min and max similarities
#         min_similarity = min(min_similarity, similarity)
#         max_similarity = max(max_similarity, similarity)
#
#         # Add to the sum for calculating the average
#         similarity_sum += similarity
#
#         # Calculate the average similarity
#     num_records = len(df['sentence1_embeddings'])
#     average_similarity = similarity_sum / num_records
#
#     print("Minimum Cosine Similarity:", min_similarity)
#     print("Maximum Cosine Similarity:", max_similarity)
#     print("Average Cosine Similarity:", average_similarity, '\n')
train_df['label'] = train_df['label'].replace(1, 0)
train_df['label'] = train_df['label'].replace(2, 1)

# Split the DataFrame based on the 'Label' column
label_values = train_df['label'].unique()  # [0 1]
dfs = {}
for label_value in label_values:
    dfs[label_value] = train_df[train_df['label'] == label_value]

non_con = dfs[0]
con = dfs[1]
for threshold in np.linspace(0.75, 1, 100):
    predictions = []
    for df in [train_df]:
        min_similarity = float('inf')
        max_similarity = float('-inf')
        similarity_sum = 0.0
        for t1, t2 in zip(df['sentence1_embeddings'], df['sentence2_embeddings']):
            t1 = t1.numpy()
            t2 = t2.numpy()

            # Reshape the tensors if needed (e.g., for 1D tensors)
            if len(t1.shape) == 1:
                t1 = t1.reshape(1, -1)
            if len(t2.shape) == 1:
                t2 = t2.reshape(1, -1)

            predictions.append(predict(t1, t2, 0.998))

        print(
            f"\tThreshold: {threshold}\n{100 * accuracy_score(train_df['label'].values, predictions):.2f}% | F1: {f1_score(train_df['label'].values, predictions, average='macro'):.4f} | P: {precision_score(train_df['label'].values, predictions, average='macro'):.4f} | R: {recall_score(train_df['label'].values, predictions, average='macro'):.4f}")
"""

if __name__ == '__main__':
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 64
    for threshold in np.arange(0.95, 1.01, 0.01):
        acc: list = []
        f1: list = []
        precision: list = []
        recall: list = []
        for i in range(30):
            for name, model_class in [
                ("Siamese", SiameseContrastiveModel()),
            ]:
                train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
                train_df['label'] = train_df['label'].replace(1, 0)
                train_df['label'] = train_df['label'].replace(2, 1)
                valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
                valid_df['label'] = valid_df['label'].replace(1, 0)
                valid_df['label'] = valid_df['label'].replace(2, 1)
                test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")
                test_df['label'] = test_df['label'].replace(1, 0)
                test_df['label'] = test_df['label'].replace(2, 1)

                model = model_class

                train_loss = model.fit(train_df, batch_size=64, num_epochs=10, device="cuda", margin=0.5, verbose=False)

                # Predict using the model
                predictions = model.predict(test_df, batch_size=64, device="cuda", similarity_threshold=threshold)
                # print(predictions)

                # Assuming you have true labels for the test data
                true_labels = test_df["label"].values  # Replace "true_labels" with the actual column name

                # model_save_path: str = f"Models/{name}.pt"
                # torch.save(model.state_dict(), model_save_path)
                # model.load_state_dict(torch.load(model_save_path))

                # unique_values, counts = np.unique(predictions, return_counts=True)
                # for value, count in zip(unique_values, counts):
                #     print(f"Class {value}: {count} predictions")

                # print(f"Accuracy:  {100 * accuracy_score(test_df['label'].values, predictions):.2f}")
                # print(f"F1-Score:  {f1_score(test_df['label'].values, predictions, average='macro'):.4f}")
                # print(f"Precision: {precision_score(test_df['label'].values, predictions, average='macro'):.4f}")
                # print(f"Recall:    {recall_score(test_df['label'].values, predictions, average='macro'):.4f}")
                acc.append(accuracy_score(test_df["label"].values, predictions))
                f1.append(f1_score(test_df["label"].values, predictions, average="macro"))
                precision.append(precision_score(test_df["label"].values, predictions, average="macro"))
                recall.append(recall_score(test_df["label"].values, predictions, average="macro"))
        print(f"\t{name} | {threshold}")
        print(
            f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
        )
