import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AutoModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from ContradictDetectNN import count_negations, ph_to_tensor
from sklearn.preprocessing import StandardScaler

from persitent_homology import persistent_homology_features

from variables import DEVICE
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from clean_dataset import create_subset_with_ratio, label_mapping
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn


def get_embeddings(df, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    embeddings1 = []
    embeddings2 = []

    for i in range(0, len(df), BATCH_SIZE):
        batch = df[i:i + BATCH_SIZE]
        sentences1 = batch["sentence1"].values.tolist()
        sentences2 = batch["sentence2"].values.tolist()

        inputs1 = tokenizer(sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True).to(
            DEVICE)
        inputs2 = tokenizer(sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True).to(
            DEVICE)

        with torch.no_grad():
            outputs1 = model(**inputs1)
            embeddings1_batch = outputs1.last_hidden_state
            embeddings1.append(torch.mean(embeddings1_batch, dim=1))

            outputs2 = model(**inputs2)
            embeddings2_batch = outputs2.last_hidden_state
            embeddings2.append(torch.mean(embeddings2_batch, dim=1))

    # Concatenate the batches into single tensors
    embeddings1 = torch.cat(embeddings1, dim=0)
    embeddings2 = torch.cat(embeddings2, dim=0)

    return embeddings1, embeddings2


def get_features(df):
    scaler = StandardScaler()
    s1_a = torch.stack(
        df["sentence1_ph_a"].values.tolist(), dim=0
    ).to(DEVICE)
    s1_b = torch.stack(
        df["sentence1_ph_b"].values.tolist(), dim=0
    ).to(DEVICE)
    s2_a = torch.stack(
        df["sentence2_ph_a"].values.tolist(), dim=0
    ).to(DEVICE)
    s2_b = torch.stack(
        df["sentence2_ph_b"].values.tolist(), dim=0
    ).to(DEVICE)
    return scaler.fit_transform(s1_a), scaler.fit_transform(s1_b), scaler.fit_transform(s2_a), scaler.fit_transform(
        s2_b)


def replace_inf(tensors: torch.Tensor) -> torch.Tensor:
    tensor_with_inf = tensors.cpu().numpy()

    for i in range(tensor_with_inf.shape[0]):
        row_values = tensor_with_inf[i, :]
        max_finite_value = np.max(row_values[np.isfinite(row_values)])

        # Check if the largest value is 3.4028235e+38
        if np.isinf(max_finite_value) and not np.isnan(max_finite_value):
            max_finite_value = np.max(row_values[row_values != 3.4028235e+38])

        tensor_with_inf[i, :] = np.where(row_values == 3.4028235e+38, max_finite_value * 2, row_values)
        # print(max_finite_value)
    return torch.tensor(tensor_with_inf)


class BBU:
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Extract logits from the outputs tuple

                # Calculate the loss
                loss = criterion(logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]

                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class BBUNeg:
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if 'negation' not in training_data.columns:
            training_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                  in
                                  training_data.iterrows()]
            training_data["negation"] = training_negations
            validation_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                    in
                                    validation_data.iterrows()]
            validation_data["negation"] = validation_negations

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_negation = torch.tensor(
                    training_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Concatenate the extra feature to the input_ids
                input_ids_with_extra_feature = torch.cat([input_ids, batch_negation.unsqueeze(1)], dim=1)

                # Concatenate the extra feature to the input_ids
                extra_feature_attention_mask = torch.ones_like(batch_negation)

                # Combine the attention_mask for input_ids and the extra feature
                combined_attention_mask = torch.cat([attention_mask, extra_feature_attention_mask.unsqueeze(1)], dim=1)

                # Forward pass up to the last layer
                outputs = self.model(input_ids_with_extra_feature, attention_mask=combined_attention_mask)
                last_hidden_state = outputs[0]  # Get the last hidden state from the outputs [batch_size, num _classes]

                # Calculate the loss
                loss = criterion(last_hidden_state, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(last_hidden_state, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())

            average_loss: float = running_loss / (len(training_data) / batch_size)

            # Calculate training accuracy and F1-score
            training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
            training_f1: float = f1_score(
                all_true_labels, all_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_negation = torch.tensor(
                        validation_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                    ).to(device)
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    ).to(device)

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1, batch_sentences2, max_length=256, return_tensors="pt", padding="max_length",
                        truncation=True
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Concatenate the extra feature to the input_ids
                    input_ids_with_extra_feature = torch.cat([input_ids, batch_negation.unsqueeze(1)], dim=1)

                    # Concatenate the extra feature to the input_ids
                    extra_feature_attention_mask = torch.ones_like(batch_negation)

                    # Combine the attention_mask for input_ids and the extra feature
                    combined_attention_mask = torch.cat([attention_mask, extra_feature_attention_mask.unsqueeze(1)],
                                                        dim=1)

                    # Forward pass up to the last layer
                    outputs = self.model(input_ids_with_extra_feature, attention_mask=combined_attention_mask)
                    last_hidden_state = outputs[
                        0]  # Get the last hidden state from the outputs [batch_size, num _classes]
                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(last_hidden_state, dim=1)  # Apply softmax to output
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()
        test_predictions = np.array([])

        if 'negation' not in test_data.columns:
            testing_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row in
                                 test_data.iterrows()]
            test_data["negation"] = testing_negations

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                batch_negation = torch.tensor(
                    test_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, max_length=256, return_tensors="pt", padding="max_length",
                    truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                # Concatenate the extra feature to the input_ids
                input_ids_with_extra_feature = torch.cat([input_ids, batch_negation.unsqueeze(1)], dim=1)
                # Concatenate the extra feature to the input_ids
                extra_feature_attention_mask = torch.ones_like(batch_negation)
                # Combine the attention_mask for input_ids and the extra feature
                combined_attention_mask = torch.cat([attention_mask, extra_feature_attention_mask.unsqueeze(1)],
                                                    dim=1)
                # Forward pass up to the last layer
                outputs = self.model(input_ids_with_extra_feature, attention_mask=combined_attention_mask)
                # Extract the last hidden state from the model's output
                last_hidden_state = outputs[0]
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(last_hidden_state, dim=1)  # Apply softmax to output
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class BBUPHSVM:
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased"):
        super(BBUPHSVM, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(self.model_name,
                                                 num_labels=self.num_classes).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(768 + 768 + 260 + 50 + 260 + 50, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        ).to(DEVICE)

        # Classification layer
        self.classification_layer = nn.Linear(256, num_classes).to(DEVICE)

    def forward(
            self,
            input_ids1,
            attention_mask1,
            batch_s1_feature_a,
            batch_s1_feature_b,
            input_ids2,
            attention_mask2,
            batch_s2_feature_a,
            batch_s2_feature_b,
    ):
        # Move input tensors to the desired device
        input_ids1 = input_ids1.to(DEVICE)
        attention_mask1 = attention_mask1.to(DEVICE)
        batch_s1_feature_a = batch_s1_feature_a.to(DEVICE)
        batch_s1_feature_b = batch_s1_feature_b.to(DEVICE)
        input_ids2 = input_ids2.to(DEVICE)
        attention_mask2 = attention_mask2.to(DEVICE)
        batch_s2_feature_a = batch_s2_feature_a.to(DEVICE)
        batch_s2_feature_b = batch_s2_feature_b.to(DEVICE)

        # BERT forward pass for sentence 1
        outputs1 = self.encoder(input_ids1, attention_mask=attention_mask1)
        embeddings1 = outputs1.last_hidden_state  # Extract BERT embeddings
        # BERT forward pass for sentence 2
        outputs2 = self.encoder(input_ids2, attention_mask=attention_mask2)
        embeddings2 = outputs2.last_hidden_state  # Extract BERT embeddings

        # Concatenate BERT embeddings and processed features
        combined_input = torch.cat(
            [embeddings1.mean(dim=1),
             embeddings2.mean(dim=1),
             batch_s1_feature_a,
             batch_s1_feature_b,
             batch_s2_feature_a,
             batch_s2_feature_b],
            dim=1,
        )

        # Pass through hidden layers
        hidden_output = self.hidden_layers(combined_input)
        # Classification
        logits = self.classification_layer(hidden_output)

        return logits.to(DEVICE)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.to(device)

        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=5e-6)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            try:
                training_data[column] = training_data[column].apply(ph_to_tensor)
                training_data[column] = training_data[column].apply(lambda x: x[:, -1])
            except AttributeError:
                continue
            try:
                # Validation cleaning
                validation_data[column] = validation_data[column].apply(ph_to_tensor)
                validation_data[column] = validation_data[column].apply(lambda x: x[:, -1])
            except AttributeError:
                continue

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Additional input vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                train_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=batch_s1_feature_a,
                    batch_s1_feature_b=batch_s1_feature_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=batch_s2_feature_a,
                    batch_s2_feature_b=batch_s2_feature_b,
                )

                # Calculate the loss
                loss = criterion(train_logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(train_logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Additional input vectors
                    s1_ph_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s1_ph_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)

                    # Tokenize and encode sentences 1 and 2 separately
                    inputs1 = self.tokenizer(
                        batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)
                    inputs2 = self.tokenizer(
                        batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)

                    input_ids1 = inputs1["input_ids"]
                    attention_mask1 = inputs1["attention_mask"]
                    input_ids2 = inputs2["input_ids"]
                    attention_mask2 = inputs2["attention_mask"]

                    # calculate logits
                    val_logits = self(
                        input_ids1=input_ids1,
                        attention_mask1=attention_mask1,
                        batch_s1_feature_a=s1_ph_a,
                        batch_s1_feature_b=s1_ph_b,
                        input_ids2=input_ids2,
                        attention_mask2=attention_mask2,
                        batch_s2_feature_a=s2_ph_a,
                        batch_s2_feature_b=s2_ph_b,
                    )

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(val_logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.to(device)
        self.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            try:
                test_data[column] = test_data[column].apply(ph_to_tensor)
                test_data[column] = test_data[column].apply(lambda x: x[:, -1])
            except AttributeError:
                continue

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                # Additional input vectors
                s1_ph_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s1_ph_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                test_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=s1_ph_a,
                    batch_s1_feature_b=s1_ph_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=s2_ph_a,
                    batch_s2_feature_b=s2_ph_b,
                )

                class_probabilities = torch.softmax(test_logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class BBUPHMLP(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased"):
        super(BBUPHMLP, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(self.model_name,
                                                 num_labels=self.num_classes).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Define MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(3692, 300),  # Input size is 1200 (max_pool1 + max_pool2 + avg_pool1 + avg_pool2)
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, num_classes)  # Output layer with num_classes
        ).to(DEVICE)

    def forward(
            self,
            input_ids1,
            attention_mask1,
            batch_s1_feature_a,
            batch_s1_feature_b,
            input_ids2,
            attention_mask2,
            batch_s2_feature_a,
            batch_s2_feature_b,
    ):
        # Move input tensors to the desired device
        input_ids1 = input_ids1.to(DEVICE)
        attention_mask1 = attention_mask1.to(DEVICE)
        batch_s1_feature_a = batch_s1_feature_a.to(DEVICE)
        batch_s1_feature_b = batch_s1_feature_b.to(DEVICE)
        input_ids2 = input_ids2.to(DEVICE)
        attention_mask2 = attention_mask2.to(DEVICE)
        batch_s2_feature_a = batch_s2_feature_a.to(DEVICE)
        batch_s2_feature_b = batch_s2_feature_b.to(DEVICE)

        # BERT forward pass for sentence 1
        # BERT forward pass for sentence 1
        outputs1 = self.encoder(input_ids1, attention_mask=attention_mask1)
        embeddings1 = outputs1.last_hidden_state  # Extract BERT embeddings

        # BERT forward pass for sentence 2
        outputs2 = self.encoder(input_ids2, attention_mask=attention_mask2)
        embeddings2 = outputs2.last_hidden_state  # Extract BERT embeddings

        # Max pooling for sentence 1 and sentence 2
        max_pool1, _ = embeddings1.max(dim=1)
        max_pool2, _ = embeddings2.max(dim=1)

        # Average pooling for sentence 1 and sentence 2
        avg_pool1 = embeddings1.mean(dim=1)
        avg_pool2 = embeddings2.mean(dim=1)

        # Concatenate max and average pooled embeddings with other features
        combined_input = torch.cat(
            [max_pool1,
             max_pool2,
             avg_pool1,
             avg_pool2,
             batch_s1_feature_a,
             batch_s1_feature_b,
             batch_s2_feature_a,
             batch_s2_feature_b],
            dim=1,
        )

        # Pass through MLP layers
        logits = self.mlp(combined_input)

        return logits.to(DEVICE)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.to(device)

        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-10)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if "sentence1_ph_a" in training_data.columns:
            for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
                try:
                    # Training cleaning
                    training_data[column] = training_data[column].apply(ph_to_tensor)
                    training_data[column] = training_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue
                try:
                    # Validation cleaning
                    validation_data[column] = validation_data[column].apply(ph_to_tensor)
                    validation_data[column] = validation_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Additional input vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                train_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=batch_s1_feature_a,
                    batch_s1_feature_b=batch_s1_feature_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=batch_s2_feature_a,
                    batch_s2_feature_b=batch_s2_feature_b,
                )

                # Calculate the loss
                loss = criterion(train_logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(train_logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Additional input vectors
                    s1_ph_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s1_ph_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)

                    # Tokenize and encode sentences 1 and 2 separately
                    inputs1 = self.tokenizer(
                        batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)
                    inputs2 = self.tokenizer(
                        batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)

                    input_ids1 = inputs1["input_ids"]
                    attention_mask1 = inputs1["attention_mask"]
                    input_ids2 = inputs2["input_ids"]
                    attention_mask2 = inputs2["attention_mask"]

                    # calculate logits
                    val_logits = self(
                        input_ids1=input_ids1,
                        attention_mask1=attention_mask1,
                        batch_s1_feature_a=s1_ph_a,
                        batch_s1_feature_b=s1_ph_b,
                        input_ids2=input_ids2,
                        attention_mask2=attention_mask2,
                        batch_s2_feature_a=s2_ph_a,
                        batch_s2_feature_b=s2_ph_b,
                    )

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(val_logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.to(device)
        self.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            try:
                test_data[column] = test_data[column].apply(ph_to_tensor)
                test_data[column] = test_data[column].apply(lambda x: x[:, 0])
            except AttributeError:
                continue

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                # Additional input vectors
                s1_ph_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s1_ph_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                test_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=s1_ph_a,
                    batch_s1_feature_b=s1_ph_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=s2_ph_a,
                    batch_s2_feature_b=s2_ph_b,
                )

                class_probabilities = torch.softmax(test_logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class BBUNegPHMLP(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased"):
        super(BBUNegPHMLP, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(self.model_name,
                                                 num_labels=self.num_classes).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Define MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(3693, 300),  # Input size is 1200 (max_pool1 + max_pool2 + avg_pool1 + avg_pool2)
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, num_classes)  # Output layer with num_classes
        ).to(DEVICE)

    def forward(
            self,
            input_ids1,
            attention_mask1,
            batch_s1_feature_a,
            batch_s1_feature_b,
            input_ids2,
            attention_mask2,
            batch_s2_feature_a,
            batch_s2_feature_b,
            batch_negation,
    ):
        # Move input tensors to the desired device
        input_ids1 = input_ids1.to(DEVICE)
        attention_mask1 = attention_mask1.to(DEVICE)
        batch_s1_feature_a = batch_s1_feature_a.to(DEVICE)
        batch_s1_feature_b = batch_s1_feature_b.to(DEVICE)
        input_ids2 = input_ids2.to(DEVICE)
        attention_mask2 = attention_mask2.to(DEVICE)
        batch_s2_feature_a = batch_s2_feature_a.to(DEVICE)
        batch_s2_feature_b = batch_s2_feature_b.to(DEVICE)
        batch_negation = batch_negation.to(DEVICE)

        # BERT forward pass for sentence 1
        # BERT forward pass for sentence 1
        outputs1 = self.encoder(input_ids1, attention_mask=attention_mask1)
        embeddings1 = outputs1.last_hidden_state  # Extract BERT embeddings

        # BERT forward pass for sentence 2
        outputs2 = self.encoder(input_ids2, attention_mask=attention_mask2)
        embeddings2 = outputs2.last_hidden_state  # Extract BERT embeddings

        # Max pooling for sentence 1 and sentence 2
        max_pool1, _ = embeddings1.max(dim=1)
        max_pool2, _ = embeddings2.max(dim=1)

        # Average pooling for sentence 1 and sentence 2
        avg_pool1 = embeddings1.mean(dim=1)
        avg_pool2 = embeddings2.mean(dim=1)

        # Concatenate max and average pooled embeddings with other features
        combined_input = torch.cat(
            [max_pool1,
             max_pool2,
             avg_pool1,
             avg_pool2,
             batch_s1_feature_a,
             batch_s1_feature_b,
             batch_s2_feature_a,
             batch_s2_feature_b,
             batch_negation],
            dim=1,
        )

        # Pass through MLP layers
        logits = self.mlp(combined_input)

        return logits.to(DEVICE)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.to(device)

        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-10)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if 'negation' not in training_data.columns:
            training_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                  in
                                  training_data.iterrows()]
            training_data["negation"] = training_negations
            validation_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                    in
                                    validation_data.iterrows()]
            validation_data["negation"] = validation_negations

        if "sentence1_ph_a" in training_data.columns:
            for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
                try:
                    # Training cleaning
                    training_data[column] = training_data[column].apply(ph_to_tensor)
                    training_data[column] = training_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue
                try:
                    # Validation cleaning
                    validation_data[column] = validation_data[column].apply(ph_to_tensor)
                    validation_data[column] = validation_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_negation = torch.tensor(
                    training_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).view(-1, 1).to(device)
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Additional input vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                train_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=batch_s1_feature_a,
                    batch_s1_feature_b=batch_s1_feature_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=batch_s2_feature_a,
                    batch_s2_feature_b=batch_s2_feature_b,
                    batch_negation=batch_negation,
                )

                # Calculate the loss
                loss = criterion(train_logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(train_logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_negation = torch.tensor(
                        validation_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                    ).view(-1, 1).to(device)
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Additional input vectors
                    s1_ph_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s1_ph_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)

                    # Tokenize and encode sentences 1 and 2 separately
                    inputs1 = self.tokenizer(
                        batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)
                    inputs2 = self.tokenizer(
                        batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)

                    input_ids1 = inputs1["input_ids"]
                    attention_mask1 = inputs1["attention_mask"]
                    input_ids2 = inputs2["input_ids"]
                    attention_mask2 = inputs2["attention_mask"]

                    # calculate logits
                    val_logits = self(
                        input_ids1=input_ids1,
                        attention_mask1=attention_mask1,
                        batch_s1_feature_a=s1_ph_a,
                        batch_s1_feature_b=s1_ph_b,
                        input_ids2=input_ids2,
                        attention_mask2=attention_mask2,
                        batch_s2_feature_a=s2_ph_a,
                        batch_s2_feature_b=s2_ph_b,
                        batch_negation=batch_negation,
                    )

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(val_logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.to(device)
        self.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        if 'negation' not in test_data.columns:
            testing_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row in
                                 test_data.iterrows()]
            test_data["negation"] = testing_negations

        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            try:
                test_data[column] = test_data[column].apply(ph_to_tensor)
                test_data[column] = test_data[column].apply(lambda x: x[:, 0])
            except AttributeError:
                continue

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                batch_negation = torch.tensor(
                    test_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).view(-1, 1).to(device)

                # Additional input vectors
                s1_ph_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s1_ph_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                test_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=s1_ph_a,
                    batch_s1_feature_b=s1_ph_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=s2_ph_a,
                    batch_s2_feature_b=s2_ph_b,
                    batch_negation=batch_negation,
                )

                class_probabilities = torch.softmax(test_logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class RoBERTaB:
    def __init__(self, num_classes: int, model_name: str = "roberta-base"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Extract logits from the outputs tuple

                # Calculate the loss
                loss = criterion(logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class RoBERTaL:
    def __init__(self, num_classes: int, model_name: str = "roberta-large"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Extract logits from the outputs tuple

                # Calculate the loss
                loss = criterion(logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class RoBERTaBNeg:
    def __init__(self, num_classes: int, model_name: str = "roberta-base"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if 'negation' not in training_data.columns:
            training_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                  in
                                  training_data.iterrows()]
            training_data["negation"] = training_negations
            validation_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                    in
                                    validation_data.iterrows()]
            validation_data["negation"] = validation_negations

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_negation = torch.tensor(
                    training_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                # Concatenate the extra feature to the input_ids
                input_ids_with_extra_feature = torch.cat([input_ids, batch_negation.unsqueeze(1)], dim=1)
                # Concatenate the extra feature to the input_ids
                extra_feature_attention_mask = torch.ones_like(batch_negation)
                # Combine the attention_mask for input_ids and the extra feature
                combined_attention_mask = torch.cat([attention_mask, extra_feature_attention_mask.unsqueeze(1)], dim=1)
                # print(input_ids.shape)  # torch.Size([64, 40])
                # print(attention_mask.shape)  # torch.Size([64, 40])
                # print(batch_negation.unsqueeze(1).shape)  # torch.Size([64, 1])
                # print(input_ids_with_extra_feature.shape)  # torch.Size([64, 41])
                # print(combined_attention_mask.shape)  # torch.Size([64, 41])
                # Forward pass up to the last layer
                outputs = self.model(input_ids_with_extra_feature, attention_mask=combined_attention_mask)
                last_hidden_state = outputs[0]  # Get the last hidden state from the outputs [batch_size, num _classes]
                # Calculate the loss
                loss = criterion(last_hidden_state, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(last_hidden_state, dim=1)
                predicted_classes = torch.argmax(class_probabilities, dim=1)
                true_labels = batch_labels.view(-1).cpu().numpy()

                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_classes.cpu().numpy())

            average_loss: float = running_loss / (len(training_data) / batch_size)

            # Calculate training accuracy and F1-score
            training_accuracy: float = accuracy_score(all_true_labels, all_predicted_labels)
            training_f1: float = f1_score(
                all_true_labels, all_predicted_labels, average="macro"
            )  # You can choose 'micro' or 'weighted' as well

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_negation = torch.tensor(
                        validation_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                    ).to(device)
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    ).to(device)

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1, batch_sentences2, max_length=256, return_tensors="pt", padding="max_length",
                        truncation=True
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Concatenate the extra feature to the input_ids
                    input_ids_with_extra_feature = torch.cat([input_ids, batch_negation.unsqueeze(1)], dim=1)

                    # Concatenate the extra feature to the input_ids
                    extra_feature_attention_mask = torch.ones_like(batch_negation)

                    # Combine the attention_mask for input_ids and the extra feature
                    combined_attention_mask = torch.cat([attention_mask, extra_feature_attention_mask.unsqueeze(1)],
                                                        dim=1)

                    # Forward pass up to the last layer
                    outputs = self.model(input_ids_with_extra_feature, attention_mask=combined_attention_mask)
                    last_hidden_state = outputs[0]
                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(last_hidden_state, dim=1)  # Apply softmax to output
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()
        test_predictions = np.array([])

        if 'negation' not in test_data.columns:
            testing_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row in
                                 test_data.iterrows()]
            test_data["negation"] = testing_negations

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                batch_negation = torch.tensor(
                    test_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                # Concatenate the extra feature to the input_ids
                input_ids_with_extra_feature = torch.cat([input_ids, batch_negation.unsqueeze(1)], dim=1)
                # Concatenate the extra feature to the input_ids
                extra_feature_attention_mask = torch.ones_like(batch_negation)
                # Combine the attention_mask for input_ids and the extra feature
                combined_attention_mask = torch.cat([attention_mask, extra_feature_attention_mask.unsqueeze(1)],
                                                    dim=1)
                # Forward pass up to the last layer
                outputs = self.model(input_ids_with_extra_feature, attention_mask=combined_attention_mask)
                # Extract the last hidden state from the model's output
                last_hidden_state = outputs[0]
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(last_hidden_state, dim=1)  # Apply softmax to output
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class RoBERTaBPHMLP(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "roberta-base"):
        super(RoBERTaBPHMLP, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(self.model_name,
                                                 num_labels=self.num_classes).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Define MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(3692, 300),  # Input size is 1200 (max_pool1 + max_pool2 + avg_pool1 + avg_pool2)
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, num_classes)  # Output layer with num_classes
        ).to(DEVICE)

    def forward(
            self,
            input_ids1,
            attention_mask1,
            batch_s1_feature_a,
            batch_s1_feature_b,
            input_ids2,
            attention_mask2,
            batch_s2_feature_a,
            batch_s2_feature_b,
    ):
        # Move input tensors to the desired device
        input_ids1 = input_ids1.to(DEVICE)
        attention_mask1 = attention_mask1.to(DEVICE)
        batch_s1_feature_a = batch_s1_feature_a.to(DEVICE)
        batch_s1_feature_b = batch_s1_feature_b.to(DEVICE)
        input_ids2 = input_ids2.to(DEVICE)
        attention_mask2 = attention_mask2.to(DEVICE)
        batch_s2_feature_a = batch_s2_feature_a.to(DEVICE)
        batch_s2_feature_b = batch_s2_feature_b.to(DEVICE)

        # BERT forward pass for sentence 1
        # BERT forward pass for sentence 1
        outputs1 = self.encoder(input_ids1, attention_mask=attention_mask1)
        embeddings1 = outputs1.last_hidden_state  # Extract BERT embeddings

        # BERT forward pass for sentence 2
        outputs2 = self.encoder(input_ids2, attention_mask=attention_mask2)
        embeddings2 = outputs2.last_hidden_state  # Extract BERT embeddings

        # Max pooling for sentence 1 and sentence 2
        max_pool1, _ = embeddings1.max(dim=1)
        max_pool2, _ = embeddings2.max(dim=1)

        # Average pooling for sentence 1 and sentence 2
        avg_pool1 = embeddings1.mean(dim=1)
        avg_pool2 = embeddings2.mean(dim=1)

        # Concatenate max and average pooled embeddings with other features
        combined_input = torch.cat(
            [max_pool1,
             max_pool2,
             avg_pool1,
             avg_pool2,
             batch_s1_feature_a,
             batch_s1_feature_b,
             batch_s2_feature_a,
             batch_s2_feature_b],
            dim=1,
        )

        # Pass through MLP layers
        logits = self.mlp(combined_input)

        return logits.to(DEVICE)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.to(device)

        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-10)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if "sentence1_ph_a" in training_data.columns:
            for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
                try:
                    # Training cleaning
                    training_data[column] = training_data[column].apply(ph_to_tensor)
                    training_data[column] = training_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue
                try:
                    # Validation cleaning
                    validation_data[column] = validation_data[column].apply(ph_to_tensor)
                    validation_data[column] = validation_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Additional input vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                train_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=batch_s1_feature_a,
                    batch_s1_feature_b=batch_s1_feature_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=batch_s2_feature_a,
                    batch_s2_feature_b=batch_s2_feature_b,
                )

                # Calculate the loss
                loss = criterion(train_logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(train_logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Additional input vectors
                    s1_ph_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s1_ph_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)

                    # Tokenize and encode sentences 1 and 2 separately
                    inputs1 = self.tokenizer(
                        batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)
                    inputs2 = self.tokenizer(
                        batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)

                    input_ids1 = inputs1["input_ids"]
                    attention_mask1 = inputs1["attention_mask"]
                    input_ids2 = inputs2["input_ids"]
                    attention_mask2 = inputs2["attention_mask"]

                    # calculate logits
                    val_logits = self(
                        input_ids1=input_ids1,
                        attention_mask1=attention_mask1,
                        batch_s1_feature_a=s1_ph_a,
                        batch_s1_feature_b=s1_ph_b,
                        input_ids2=input_ids2,
                        attention_mask2=attention_mask2,
                        batch_s2_feature_a=s2_ph_a,
                        batch_s2_feature_b=s2_ph_b,
                    )

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(val_logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.to(device)
        self.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            try:
                test_data[column] = test_data[column].apply(ph_to_tensor)
                test_data[column] = test_data[column].apply(lambda x: x[:, 0])
            except AttributeError:
                continue

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                # Additional input vectors
                s1_ph_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s1_ph_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                test_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=s1_ph_a,
                    batch_s1_feature_b=s1_ph_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=s2_ph_a,
                    batch_s2_feature_b=s2_ph_b,
                )

                class_probabilities = torch.softmax(test_logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class RoBERTaBNegPHMLP(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "roberta-base"):
        super(RoBERTaBNegPHMLP, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(self.model_name,
                                                 num_labels=self.num_classes).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Define MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(3693, 300),  # Input size is 1200 (max_pool1 + max_pool2 + avg_pool1 + avg_pool2)
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, num_classes)  # Output layer with num_classes
        ).to(DEVICE)

    def forward(
            self,
            input_ids1,
            attention_mask1,
            batch_s1_feature_a,
            batch_s1_feature_b,
            input_ids2,
            attention_mask2,
            batch_s2_feature_a,
            batch_s2_feature_b,
            batch_negation,
    ):
        # Move input tensors to the desired device
        input_ids1 = input_ids1.to(DEVICE)
        attention_mask1 = attention_mask1.to(DEVICE)
        batch_s1_feature_a = batch_s1_feature_a.to(DEVICE)
        batch_s1_feature_b = batch_s1_feature_b.to(DEVICE)
        input_ids2 = input_ids2.to(DEVICE)
        attention_mask2 = attention_mask2.to(DEVICE)
        batch_s2_feature_a = batch_s2_feature_a.to(DEVICE)
        batch_s2_feature_b = batch_s2_feature_b.to(DEVICE)
        batch_negation = batch_negation.to(DEVICE)

        # BERT forward pass for sentence 1
        # BERT forward pass for sentence 1
        outputs1 = self.encoder(input_ids1, attention_mask=attention_mask1)
        embeddings1 = outputs1.last_hidden_state  # Extract BERT embeddings

        # BERT forward pass for sentence 2
        outputs2 = self.encoder(input_ids2, attention_mask=attention_mask2)
        embeddings2 = outputs2.last_hidden_state  # Extract BERT embeddings

        # Max pooling for sentence 1 and sentence 2
        max_pool1, _ = embeddings1.max(dim=1)
        max_pool2, _ = embeddings2.max(dim=1)

        # Average pooling for sentence 1 and sentence 2
        avg_pool1 = embeddings1.mean(dim=1)
        avg_pool2 = embeddings2.mean(dim=1)

        # Concatenate max and average pooled embeddings with other features
        combined_input = torch.cat(
            [max_pool1,
             max_pool2,
             avg_pool1,
             avg_pool2,
             batch_s1_feature_a,
             batch_s1_feature_b,
             batch_s2_feature_a,
             batch_s2_feature_b,
             batch_negation],
            dim=1,
        )

        # Pass through MLP layers
        logits = self.mlp(combined_input)

        return logits.to(DEVICE)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.to(device)

        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-10)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if 'negation' not in training_data.columns:
            training_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                  in
                                  training_data.iterrows()]
            training_data["negation"] = training_negations
            validation_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row
                                    in
                                    validation_data.iterrows()]
            validation_data["negation"] = validation_negations

        if "sentence1_ph_a" in training_data.columns:
            for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
                try:
                    # Training cleaning
                    training_data[column] = training_data[column].apply(ph_to_tensor)
                    training_data[column] = training_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue
                try:
                    # Validation cleaning
                    validation_data[column] = validation_data[column].apply(ph_to_tensor)
                    validation_data[column] = validation_data[column].apply(lambda x: x[:, 0])
                except AttributeError:
                    continue

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_negation = torch.tensor(
                    training_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).view(-1, 1).to(device)
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Additional input vectors
                batch_s1_feature_a = torch.stack(
                    training_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s1_feature_b = torch.stack(
                    training_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_a = torch.stack(
                    training_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                batch_s2_feature_b = torch.stack(
                    training_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                train_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=batch_s1_feature_a,
                    batch_s1_feature_b=batch_s1_feature_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=batch_s2_feature_a,
                    batch_s2_feature_b=batch_s2_feature_b,
                    batch_negation=batch_negation,
                )

                # Calculate the loss
                loss = criterion(train_logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(train_logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_negation = torch.tensor(
                        validation_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                    ).view(-1, 1).to(device)
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Additional input vectors
                    s1_ph_a = torch.stack(
                        validation_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s1_ph_b = torch.stack(
                        validation_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_a = torch.stack(
                        validation_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)
                    s2_ph_b = torch.stack(
                        validation_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                    ).to(device)

                    # Tokenize and encode sentences 1 and 2 separately
                    inputs1 = self.tokenizer(
                        batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)
                    inputs2 = self.tokenizer(
                        batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                    ).to(device)

                    input_ids1 = inputs1["input_ids"]
                    attention_mask1 = inputs1["attention_mask"]
                    input_ids2 = inputs2["input_ids"]
                    attention_mask2 = inputs2["attention_mask"]

                    # calculate logits
                    val_logits = self(
                        input_ids1=input_ids1,
                        attention_mask1=attention_mask1,
                        batch_s1_feature_a=s1_ph_a,
                        batch_s1_feature_b=s1_ph_b,
                        input_ids2=input_ids2,
                        attention_mask2=attention_mask2,
                        batch_s2_feature_a=s2_ph_a,
                        batch_s2_feature_b=s2_ph_b,
                        batch_negation=batch_negation,
                    )

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(val_logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.to(device)
        self.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        if 'negation' not in test_data.columns:
            testing_negations = [count_negations([row["sentence1"].strip(), row["sentence2"].strip()]) for index, row in
                                 test_data.iterrows()]
            test_data["negation"] = testing_negations

        for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
            # Training cleaning
            try:
                test_data[column] = test_data[column].apply(ph_to_tensor)
                test_data[column] = test_data[column].apply(lambda x: x[:, 0])
            except AttributeError:
                continue

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                batch_negation = torch.tensor(
                    test_data["negation"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).view(-1, 1).to(device)

                # Additional input vectors
                s1_ph_a = torch.stack(
                    test_data["sentence1_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s1_ph_b = torch.stack(
                    test_data["sentence1_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_a = torch.stack(
                    test_data["sentence2_ph_a"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)
                s2_ph_b = torch.stack(
                    test_data["sentence2_ph_b"].values.tolist()[i: i + batch_size], dim=0
                ).to(device)

                # Tokenize and encode sentences 1 and 2 separately
                inputs1 = self.tokenizer(
                    batch_sentences1, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)
                inputs2 = self.tokenizer(
                    batch_sentences2, max_length=128, return_tensors="pt", padding='max_length', truncation=True
                ).to(device)

                input_ids1 = inputs1["input_ids"]
                attention_mask1 = inputs1["attention_mask"]
                input_ids2 = inputs2["input_ids"]
                attention_mask2 = inputs2["attention_mask"]

                # calculate logits
                test_logits = self(
                    input_ids1=input_ids1,
                    attention_mask1=attention_mask1,
                    batch_s1_feature_a=s1_ph_a,
                    batch_s1_feature_b=s1_ph_b,
                    input_ids2=input_ids2,
                    attention_mask2=attention_mask2,
                    batch_s2_feature_a=s2_ph_a,
                    batch_s2_feature_b=s2_ph_b,
                    batch_negation=batch_negation,
                )

                class_probabilities = torch.softmax(test_logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class SeqT5:
    def __init__(self, num_classes: int, model_name: str = "t5-base"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Extract logits from the outputs tuple

                # Calculate the loss
                loss = criterion(logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


class SeqGPT2:
    def __init__(self, num_classes: int, model_name: str = "gpt2"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            training_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            batch_size: int,
            num_epochs: int,
            device: str,
            verbose: bool = False,
    ) -> None:
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # train_accuracy_values: list = []
        # train_f1_values: list = []
        # val_accuracy_values: list = []
        # val_f1_values: list = []
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    training_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    training_data["sentence2"].values.tolist()[i: i + batch_size]]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                ).to(device)

                # Tokenize and encode the batch
                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Extract logits from the outputs tuple

                # Calculate the loss
                loss = criterion(logits, batch_labels)
                running_loss += loss.item()

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to class predictions
                class_probabilities = torch.softmax(logits, dim=1)  # Apply softmax to logits
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

            if verbose:
                print(
                    f"\tTraining   | Accuracy: {training_accuracy:.4f}, F1 Score: {training_f1:.4f}, Loss: {average_loss:.4f}"
                )

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for i in range(0, len(validation_data), batch_size):
                    batch_sentences1 = [str(sentence) for sentence in
                                        validation_data["sentence1"].values.tolist()[i: i + batch_size]]
                    batch_sentences2 = [str(sentence) for sentence in
                                        validation_data["sentence2"].values.tolist()[i: i + batch_size]]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i: i + batch_size], dtype=torch.long
                    )

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                    # Convert outputs to class predictions
                    class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                    val_predicted_classes = torch.argmax(class_probabilities, dim=1)

                    # extend bookkeeping lists
                    val_predictions.extend(val_predicted_classes.cpu().numpy())
                    val_true_labels.extend(batch_labels.view(-1).cpu().numpy())

            # Calculate validation accuracy and F1-score
            val_accuracy: float = accuracy_score(val_true_labels, val_predictions)
            val_f1: float = f1_score(
                val_true_labels, val_predictions, average="macro"
            )  # You can choose 'micro' or 'weighted' as well
            if verbose:
                print(f"\tValidation | Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        test_predictions = np.array([])

        with torch.no_grad():  # Disable gradient tracking during testing
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [str(sentence) for sentence in
                                    test_data["sentence1"].values.tolist()[i: i + batch_size]]
                batch_sentences2 = [str(sentence) for sentence in
                                    test_data["sentence2"].values.tolist()[i: i + batch_size]]

                inputs = self.tokenizer(
                    batch_sentences1, batch_sentences2, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                class_probabilities = torch.softmax(outputs.logits, dim=1)  # Apply softmax to logits
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions


if __name__ == "__main__":
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 16
    NUM_CLASSES: int = 3
    for name, model in [
        # ("BBU", BBU(NUM_CLASSES)),
        ('RoBERTaB', RoBERTaB(NUM_CLASSES)),
        # ("BBUNeg", BBUNeg(NUM_CLASSES)),
        # ("BBUPHMLP", BBUPHMLP(NUM_CLASSES)),
        # ('BBUNegPHMLP', BBUNegPHMLP(NUM_CLASSES)),
        # ('RoBERTaBPHMLP', RoBERTaBPHMLP(NUM_CLASSES)),
        # ('RoBERTaBNegPHMLP', RoBERTaBNegPHMLP(NUM_CLASSES)),
        # ('RoBERTaBNeg', RoBERTaBNeg(NUM_CLASSES)),
        ("RoBERTaL", RoBERTaL(NUM_CLASSES)),
        # ("SeqT5", SeqT5(NUM_CLASSES)),
        # ('SeqGPT2', SeqGPT2(NUM_CLASSES)),
    ]:
        for dataset_percentage in [0.1]:
            print(name)
            acc = []
            f1 = []
            precision = []
            recall = []
            train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
            valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
            test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")
            # train_df = create_subset_with_ratio(pd.read_csv("Data/SNLI/train_cleaned.csv"), dataset_percentage,
            #                                     'gold_label')
            # valid_df = pd.read_csv("Data/SNLI/valid_cleaned.csv")
            # test_df = pd.read_csv("Data/SNLI/test_cleaned.csv")
            # train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
            # valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
            # test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")
            for dataset in ['mismatch']:
                # train_df = label_mapping(
                #     df=pd.read_csv("Data/MultiNLI/train_cleaned_subset.csv"),
                #     from_col='gold_label',
                #     to_col='label')
                # valid_df = label_mapping(pd.read_csv(f"Data/MultiNLI/{dataset}_cleaned.csv"))
                # test_df = pd.read_csv(f"Data/MultiNLI/test_{dataset}_cleaned.csv")
                for i in range(1):
                    print(f"{name} -> {dataset} Started")
                    sentenceBERT = model
                    start_time = time.time()
                    # try:
                    sentenceBERT.fit(
                        training_data=train_df,
                        validation_data=valid_df,
                        batch_size=BATCH_SIZE,
                        num_epochs=NUM_EPOCHS,
                        device=DEVICE,
                        verbose=False,
                    )
                    # except torch.cuda.OutOfMemoryError:
                    #     print("CPU Iteration")
                    #     sentenceBERT.fit(
                    #         training_data=train_df,
                    #         validation_data=valid_df,
                    #         batch_size=BATCH_SIZE,
                    #         num_epochs=NUM_EPOCHS,
                    #         device='cpu',
                    #         verbose=True,
                    #     )
                    elapsed_time = time.time() - start_time
                    print(f"{name} -> {dataset} Trained")
                    predictions: np.ndarray = sentenceBERT.predict(test_data=test_df, batch_size=BATCH_SIZE,
                                                                   device=DEVICE)
                    del sentenceBERT
                    torch.cuda.empty_cache()
                    if 'label' in test_df.columns:
                        final_labels: np.ndarray = test_df["label"].values

                        test_accuracy = accuracy_score(final_labels, predictions)
                        test_precision = precision_score(final_labels, predictions, average="weighted")
                        test_recall = recall_score(final_labels, predictions, average="weighted")
                        test_f1 = f1_score(final_labels, predictions, average="weighted")

                        acc.append(test_accuracy)
                        f1.append(test_f1)
                        precision.append(test_precision)
                        recall.append(test_recall)
                    # else:
                    #     output_df: pd.DataFrame = pd.DataFrame({
                    #         'pairID': test_df['pairID'],
                    #         'gold_label': predictions,
                    #     })
                    #
                    #     output_df = label_mapping(output_df, 'gold_label', 'gold_label', False)
                    #
                    #     output_df.to_csv(f"Data/MultiNLI/{name}_{dataset}.csv", index=False)
                    print(f"{name} -> {dataset} Saved")
                    print(f"Iteration {i + 1} took {elapsed_time:.2f} seconds")
                #
                # print(f"\t{name} Average | {dataset_percentage * 100}% of original training data")
                # print(
                #     f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | "
                #     f"P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
                # )
