import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from torch.utils.data import Dataset, DataLoader

from ContradictDetectNN import str_to_tensor
from persitent_homology import persistent_homology_features
from variables import DEVICE


# Step 1: Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "sentence1": self.data.loc[idx, "sentence1_embeddings"],
            "sentence2": self.data.loc[idx, "sentence2_embeddings"],
        }
        return sample


class SentenceClassifier(nn.Module):
    def __init__(self):
        super(SentenceClassifier, self).__init__()

        # Define your layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.avgpool = nn.AvgPool1d(kernel_size=1)  # Add average pooling

    def forward(self, inputs):
        x1, x2, _ = inputs  # We don't need num_negation in this version

        # Apply convolution and max pooling
        x1_conv = F.relu(self.conv1(x1.unsqueeze(2)))
        x2_conv = F.relu(self.conv1(x2.unsqueeze(2)))

        # Apply max and average pooling
        x1_maxpool = self.maxpool(x1_conv)
        x2_maxpool = self.maxpool(x2_conv)
        x1_avgpool = self.avgpool(x1_conv)
        x2_avgpool = self.avgpool(x2_conv)

        # Apply tanh activation function
        x1_maxpool = torch.tanh(x1_maxpool)
        x2_maxpool = torch.tanh(x2_maxpool)
        x1_avgpool = torch.tanh(x1_avgpool)
        x2_avgpool = torch.tanh(x2_avgpool)

        # Flatten the pooled outputs
        x1_maxpool = x1_maxpool.view(x1_maxpool.size(0), -1)
        x2_maxpool = x2_maxpool.view(x2_maxpool.size(0), -1)
        x1_avgpool = x1_avgpool.view(x1_avgpool.size(0), -1)
        x2_avgpool = x2_avgpool.view(x2_avgpool.size(0), -1)

        # Concatenate the flattened max and average pooled outputs
        concatenated_maxpool = torch.cat((x1_maxpool, x2_maxpool), dim=1)  # [2, 256]
        concatenated_avgpool = torch.cat((x1_avgpool, x2_avgpool), dim=1)  # [2, 256]
        # print(f"Max Pool: {concatenated_maxpool.shape}")
        # print(f"Avg Pool: {concatenated_avgpool.shape}")

        return concatenated_maxpool, concatenated_avgpool


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        input_size = 2373  # Update this value based on your input dimensions
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 300)
        self.fc3 = nn.Linear(300, 300)
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(300, 1)  # Output layer for binary classification

    def forward(self, max_pooled, avg_pooled, num_negation, x1_a, x1_b, x2_a, x2_b):
        # convert all tensors to float32
        max_pooled = max_pooled.to(torch.float32)
        avg_pooled = avg_pooled.to(torch.float32)
        num_negation = num_negation.to(torch.float32)
        x1_a = x1_a.to(torch.float32)
        x1_b = x1_b.to(torch.float32)
        x2_a = x2_a.to(torch.float32)
        x2_b = x2_b.to(torch.float32)
        # print(max_pooled.shape)  # [batch_size, 256]
        # print(avg_pooled.shape)  # [batch_size, 256]
        # print(num_negation.shape)  # [batch_size, 1]
        # print(x1_a.shape)  # [batch_size, 260, 3]
        # print(x1_b.shape)  # [batch_size, 50, 3]
        # print(x2_a.shape)  # [batch_size, 260, 3]
        # print(x2_b.shape)  # [batch_size, 50, 3]
        # Flatten the individual inputs
        x1_a = x1_a.view(x1_a.size(0), -1)
        x1_b = x1_b.view(x1_b.size(0), -1)
        x2_a = x2_a.view(x2_a.size(0), -1)
        x2_b = x2_b.view(x2_b.size(0), -1)
        # print(x1_a.shape)  # [batch_size, 780]
        # print(x1_b.shape)  # [batch_size, 150]
        # print(x2_a.shape)  # [batch_size, 780]
        # print(x2_b.shape)  # [batch_size, 150]
        # Concatenate all inputs
        concatenated_inputs = torch.cat((max_pooled, avg_pooled, num_negation, x1_a, x1_b, x2_a, x2_b), dim=1)

        x = F.relu(self.fc1(concatenated_inputs))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        output = self.fc4(x)  # Output is a single value for binary classification

        return output


class CombinedModel(nn.Module):
    def __init__(self, sentence_classifier, mlp_model):
        super(CombinedModel, self).__init__()
        self.sentence_classifier = sentence_classifier
        self.mlp_model = mlp_model

    def forward(self, x1_input, x2_input, num_negation_input, x1_a_input, x1_b_input, x2_a_input, x2_b_input):
        inputs_sentence_classifier = [x1_input, x2_input, num_negation_input]
        # Get max and avg pooled results from the SentenceClassifier
        max_pooled, avg_pooled = self.sentence_classifier(inputs_sentence_classifier)
        # Concatenate additional inputs
        combined_additional_inputs = torch.cat((x1_a_input, x1_b_input, x2_a_input, x2_b_input), dim=1)
        # Pass max and avg pooled results and additional inputs to the MLP model
        mlp_output = self.mlp_model(
            max_pooled, avg_pooled, num_negation_input, x1_a_input, x1_b_input, x2_a_input, x2_b_input
        )
        return mlp_output


if __name__ == "__main__":
    num_epochs = 10
    batch_size = 2

    n = 20
    v = 2
    t = 2
    if n is not None and v is not None and t is not None:
        train_df = pd.read_csv("Data/match_cleaned.csv").head(n)
        valid_df = pd.read_csv("Data/mismatch_cleaned.csv").head(v)
        test_df = pd.read_csv("Data/contradiction-dataset_cleaned.csv").head(t)
    else:
        train_df = pd.read_csv("Data/match_cleaned.csv")
        valid_df = pd.read_csv("Data/mismatch_cleaned.csv")
        test_df = pd.read_csv("Data/contradiction-dataset_cleaned.csv")

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

    input_size_sentence = 513  # Update with the appropriate input size based on SentenceClassifier's output
    input_size_x1 = 3  # Update with the appropriate input size for x1
    input_size_x2 = 3  # Update with the appropriate input size for x2
    input_size_num_negation = 1  # Update with the appropriate input size for num_negation
    hidden_size = 300
    dropout_rate = 0.1

    print(f"Model on {DEVICE}")

    # initialize models
    sentence_classifier = SentenceClassifier().to(DEVICE)
    mlp_model = MLPModel().to(DEVICE)
    combined_model = CombinedModel(sentence_classifier, mlp_model).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

    # Training loop for the combined model
    for epoch in range(num_epochs):
        combined_model.train()  # Set the combined model to training mode
        running_loss = 0.0
        all_predicted_labels = []
        all_true_labels = []

        for i in range(0, len(train_df), batch_size):
            # Prepare the batch
            s1_embedding = train_bert_embeddings_sentence1[i : i + batch_size]
            s2_embedding = train_bert_embeddings_sentence2[i : i + batch_size]
            batch_labels = train_df["label"].iloc[i : i + batch_size].values
            batch_labels = torch.tensor(batch_labels.astype(float), dtype=torch.float32).view(-1, 1)
            num_negations = train_df["negation"].iloc[i : i + batch_size].values
            batch_negations = torch.tensor(num_negations.astype(float), dtype=torch.float32).view(-1, 1)

            s1_ph_features = persistent_homology_features(list(train_df["sentence1"].iloc[i : i + batch_size]))
            dim_0_s1_features = [item[0] for item in s1_ph_features]
            dim_1_s1_features = [item[1] for item in s1_ph_features]
            batch_s1_feature_a = torch.tensor(np.array(dim_0_s1_features)).to(DEVICE)
            batch_s1_feature_b = torch.tensor(np.array(dim_1_s1_features)).to(DEVICE)
            # print(batch_s1_feature_a.shape)  # [batch_size, 260, 3]
            # print(batch_s1_feature_b.shape)  # [batch_size, 50, 3]

            s2_ph_features = persistent_homology_features(list(train_df["sentence2"].iloc[i : i + batch_size]))
            dim_0_s2_features = [item[0] for item in s2_ph_features]
            dim_1_s2_features = [item[1] for item in s2_ph_features]
            batch_s2_feature_a = torch.tensor(np.array(dim_0_s2_features)).to(DEVICE)
            batch_s2_feature_b = torch.tensor(np.array(dim_1_s2_features)).to(DEVICE)
            # print(batch_s2_feature_a.shape)  # [batch_size, 260, 3]
            # print(batch_s2_feature_b.shape)  # [batch_size, 50, 3]

            # Move tensors to the DEVICE
            s1_embedding = s1_embedding.to(DEVICE)
            s2_embedding = s2_embedding.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            batch_negations = batch_negations.to(DEVICE)

            # Forward pass through the combined model
            outputs = combined_model(
                s1_embedding,
                s2_embedding,
                batch_negations,
                batch_s1_feature_a,
                batch_s1_feature_b,
                batch_s2_feature_a,
                batch_s2_feature_b,
            )

            # Compute the loss
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Convert outputs to binary predictions (0 or 1)
            predicted_labels = (outputs >= 0.5).float().view(-1).cpu().numpy()
            true_labels = batch_labels.view(-1).cpu().numpy()
            all_predicted_labels.extend(predicted_labels)
            all_true_labels.extend(true_labels)

        # Calculate training accuracy and F1-score
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        f1 = f1_score(all_true_labels, all_predicted_labels)

        # Print training metrics for this epoch
        average_loss = running_loss / (len(train_df) / batch_size)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}"
        )
    # # Inference using SentenceClassifier
    # sentence_classifier.eval()  # Set the model to evaluation mode
    #
    # with torch.no_grad():
    #     for batch_data in test_dataloader:
    #         inputs = batch_data  # Adjust as needed
    #         outputs = sentence_classifier(inputs)
    #         max_pooled = torch.max(outputs, dim=1)[0]
    #         avg_pooled = torch.mean(outputs, dim=1)
    #
    #         # Now use max_pooled and avg_pooled as inputs for the MLP model
    #         mlp_output = mlp_model(max_pooled, avg_pooled, x1_input, x2_input, num_negation_input)
    #         # Process the mlp_output further or calculate the loss, and update the MLP model
    #
    #         # Example backward pass and update for the MLP model
    #         optimizer_mlp.zero_grad()
    #         mlp_loss = criterion(mlp_output, mlp_targets)  # Adjust as needed
    #         mlp_loss.backward()
    #         optimizer_mlp.step()
