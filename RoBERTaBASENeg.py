import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clean_dataset import count_negations


class RoBERTaBNeg:
    def __init__(self, num_classes: int = 3, model_name: str = "roberta-base"):
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

        if "negation" not in training_data.columns:
            training_negations = [
                count_negations([row["sentence1"].strip(), row["sentence2"].strip()])
                for index, row in training_data.iterrows()
            ]
            training_data["negation"] = training_negations
            validation_negations = [
                count_negations([row["sentence1"].strip(), row["sentence2"].strip()])
                for index, row in validation_data.iterrows()
            ]
            validation_data["negation"] = validation_negations

        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch: [ {epoch} / {num_epochs} ]")
            self.model.train()  # Set the model to training mode
            running_loss: float = 0.0
            all_true_labels: list = []
            all_predicted_labels: list = []
            for i in range(0, len(training_data), batch_size):
                batch_sentences1 = [
                    str(sentence) for sentence in training_data["sentence1"].values.tolist()[i : i + batch_size]
                ]
                batch_sentences2 = [
                    str(sentence) for sentence in training_data["sentence2"].values.tolist()[i : i + batch_size]
                ]
                batch_negation = torch.tensor(
                    training_data["negation"].values.tolist()[i : i + batch_size], dtype=torch.long
                ).to(device)
                batch_labels = torch.tensor(
                    training_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
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
                    batch_sentences1 = [
                        str(sentence) for sentence in validation_data["sentence1"].values.tolist()[i : i + batch_size]
                    ]
                    batch_sentences2 = [
                        str(sentence) for sentence in validation_data["sentence2"].values.tolist()[i : i + batch_size]
                    ]
                    batch_negation = torch.tensor(
                        validation_data["negation"].values.tolist()[i : i + batch_size], dtype=torch.long
                    ).to(device)
                    batch_labels = torch.tensor(
                        validation_data["label"].values.tolist()[i : i + batch_size], dtype=torch.long
                    ).to(device)

                    # Tokenize and encode the batch
                    inputs = self.tokenizer(
                        batch_sentences1,
                        batch_sentences2,
                        max_length=256,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Concatenate the extra feature to the input_ids
                    input_ids_with_extra_feature = torch.cat([input_ids, batch_negation.unsqueeze(1)], dim=1)

                    # Concatenate the extra feature to the input_ids
                    extra_feature_attention_mask = torch.ones_like(batch_negation)

                    # Combine the attention_mask for input_ids and the extra feature
                    combined_attention_mask = torch.cat(
                        [attention_mask, extra_feature_attention_mask.unsqueeze(1)], dim=1
                    )

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

    def predict(self, test_phrase: str, proposed_reply: str, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()
        test_predictions = np.array([])

        negation_word_count: int = count_negations([test_phrase, proposed_reply])

        with torch.no_grad():
            batch_negation = torch.tensor(negation_word_count, dtype=torch.long).to(device)
            # Tokenize and encode the batch
            inputs = self.tokenizer(test_phrase, proposed_reply, return_tensors="pt", padding=True, truncation=True).to(
                device
            )
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
            # Extract the last hidden state from the model's output
            last_hidden_state = outputs[0]
            # Convert outputs to class predictions
            class_probabilities = torch.softmax(last_hidden_state, dim=1)  # Apply softmax to output
            predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()
            print(predicted_classes)

            test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions
