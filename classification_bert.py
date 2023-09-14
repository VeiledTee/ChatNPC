import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from variables import DEVICE
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SeqBBU:
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train_model(
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
                batch_sentences1 = training_data["sentence1"].values.tolist()[i : i + batch_size]
                batch_sentences2 = training_data["sentence2"].values.tolist()[i : i + batch_size]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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

            average_loss: float = running_loss / (len(train_df) / batch_size)

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
                    batch_sentences1 = validation_data["sentence1"].values.tolist()[i : i + batch_size]
                    batch_sentences2 = validation_data["sentence2"].values.tolist()[i : i + batch_size]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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
                batch_sentences1 = test_data["sentence1"].values.tolist()[i : i + batch_size]
                batch_sentences2 = test_data["sentence2"].values.tolist()[i : i + batch_size]
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


class SeqRoBERTa:
    def __init__(self, num_classes: int, model_name: str = "roberta-base"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train_model(
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
                batch_sentences1 = training_data["sentence1"].values.tolist()[i : i + batch_size]
                batch_sentences2 = training_data["sentence2"].values.tolist()[i : i + batch_size]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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

            average_loss: float = running_loss / (len(train_df) / batch_size)

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
                    batch_sentences1 = validation_data["sentence1"].values.tolist()[i : i + batch_size]
                    batch_sentences2 = validation_data["sentence2"].values.tolist()[i : i + batch_size]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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
                batch_sentences1 = test_data["sentence1"].values.tolist()[i: i + batch_size]
                batch_sentences2 = test_data["sentence2"].values.tolist()[i: i + batch_size]

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


class SeqT5:
    def __init__(self, num_classes: int, model_name: str = "t5-base"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train_model(
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
                batch_sentences1 = training_data["sentence1"].values.tolist()[i : i + batch_size]
                batch_sentences2 = training_data["sentence2"].values.tolist()[i : i + batch_size]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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

            average_loss: float = running_loss / (len(train_df) / batch_size)

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
                    batch_sentences1 = validation_data["sentence1"].values.tolist()[i : i + batch_size]
                    batch_sentences2 = validation_data["sentence2"].values.tolist()[i : i + batch_size]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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
                batch_sentences1 = test_data["sentence1"].values.tolist()[i: i + batch_size]
                batch_sentences2 = test_data["sentence2"].values.tolist()[i: i + batch_size]

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

    def train_model(
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
                batch_sentences1 = training_data["sentence1"].values.tolist()[i : i + batch_size]
                batch_sentences2 = training_data["sentence2"].values.tolist()[i : i + batch_size]
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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

            average_loss: float = running_loss / (len(train_df) / batch_size)

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
                    batch_sentences1 = validation_data["sentence1"].values.tolist()[i : i + batch_size]
                    batch_sentences2 = validation_data["sentence2"].values.tolist()[i : i + batch_size]
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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
                batch_sentences1 = test_data["sentence1"].values.tolist()[i: i + batch_size]
                batch_sentences2 = test_data["sentence2"].values.tolist()[i: i + batch_size]

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
    BATCH_SIZE: int = 64
    NUM_CLASSES: int = 3
    for name, model in [
        ("SeqBBU", SeqBBU(NUM_CLASSES)),
        ('SeqRoBERTa', SeqRoBERTa(NUM_CLASSES)),
        # ("SeqT5", SeqT5(NUM_CLASSES)),
        # ('SeqGPT2', SeqGPT2(NUM_CLASSES)),
    ]:
        acc = []
        f1 = []
        precision = []
        recall = []
        for i in range(30):
            train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
            valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
            test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")

            sequenceBERT = model
            sequenceBERT.train_model(
                training_data=train_df,
                validation_data=valid_df,
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                device=DEVICE,
                verbose=False,
            )
            predictions: np.ndarray = sequenceBERT.predict(test_data=test_df, batch_size=BATCH_SIZE, device=DEVICE)
            final_labels: np.ndarray = test_df["label"].values

            test_accuracy = accuracy_score(final_labels, predictions)
            test_precision = precision_score(final_labels, predictions, average="weighted")
            test_recall = recall_score(final_labels, predictions, average="weighted")
            test_f1 = f1_score(final_labels, predictions, average="weighted")

            acc.append(test_accuracy)
            f1.append(test_f1)
            precision.append(test_precision)
            recall.append(test_recall)
            # print(f"\tRun {i}")
            # print(f"Test Accuracy: {test_accuracy:.4f}")
            # print(f"Test F1-score: {test_f1:.4f}")
            # print(f"Test Precision: {test_precision:.4f}")
            # print(f"Test Recall: {test_recall:.4f}")
        print(f"\t{name} Average")
        print(f"Test Accuracy: {sum(acc) / len(acc):.4f}")
        print(f"Test F1-score: {sum(f1) / len(f1):.4f}")
        print(f"Test Precision: {sum(precision) / len(precision):.4f}")
        print(f"Test Recall: {sum(recall) / len(recall):.4f}")
