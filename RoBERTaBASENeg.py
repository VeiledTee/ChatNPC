import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clean_dataset import count_negations, create_subset_with_ratio
from variables import DEVICE


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

    def predict(self, test_data: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
        self.model.to(device)
        self.model.eval()
        test_predictions = np.array([])

        if "negation" not in test_data.columns:
            testing_negations = [
                count_negations([row["sentence1"].strip(), row["sentence2"].strip()])
                for index, row in test_data.iterrows()
            ]
            test_data["negation"] = testing_negations

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_sentences1 = [
                    str(sentence) for sentence in test_data["sentence1"].values.tolist()[i : i + batch_size]
                ]
                batch_sentences2 = [
                    str(sentence) for sentence in test_data["sentence2"].values.tolist()[i : i + batch_size]
                ]

                batch_negation = torch.tensor(
                    test_data["negation"].values.tolist()[i : i + batch_size], dtype=torch.long
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
                # Extract the last hidden state from the model's output
                last_hidden_state = outputs[0]
                # Convert outputs to class predictions
                class_probabilities = torch.softmax(last_hidden_state, dim=1)  # Apply softmax to output
                predicted_classes = torch.argmax(class_probabilities, dim=1).cpu().numpy()

                test_predictions = np.append(test_predictions, predicted_classes)

        return test_predictions

    def save_model(self, directory_path: str):
        # Ensure the model is in evaluation mode before saving
        self.model.eval()
        # Save the model's weights and configuration
        self.model.save_pretrained(directory_path)


if __name__ == "__main__":
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 16
    NUM_CLASSES: int = 3
    for name, model in [("RoBERTaBNeg", RoBERTaBNeg(NUM_CLASSES))]:
        for dataset_percentage in [0.1]:
            print(name)
            acc = []
            f1 = []
            precision = []
            recall = []
            # train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned.csv")
            # valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned.csv")
            test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned.csv")
            train_df = pd.concat(
                [
                    pd.read_csv("Data/SemEval2014T1/train_cleaned.csv"),
                    create_subset_with_ratio(
                        pd.read_csv("Data/SNLI/train_cleaned.csv"), dataset_percentage, "gold_label"
                    ),
                ]
            ).reset_index()
            valid_df = pd.concat(
                [pd.read_csv("Data/SemEval2014T1/valid_cleaned.csv"), pd.read_csv("Data/SNLI/valid_cleaned.csv")]
            ).reset_index()
            test_df = pd.concat(
                [pd.read_csv("Data/SemEval2014T1/test_cleaned.csv"), pd.read_csv("Data/SNLI/test_cleaned.csv")]
            ).reset_index()
            # test_df = pd.read_csv("Data/SNLI/test_cleaned.csv")
            # train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
            # valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
            # test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")
            # for dataset in ['mismatch']:
            # train_df = label_mapping(
            #     df=pd.read_csv("Data/MultiNLI/train_cleaned_subset.csv"),
            #     from_col='gold_label',
            #     to_col='label')
            # valid_df = label_mapping(pd.read_csv(f"Data/MultiNLI/{dataset}_cleaned.csv"))
            # test_df = pd.read_csv(f"Data/MultiNLI/test_{dataset}_cleaned.csv")
            for i in range(3):
                sentenceBERT = model
                start_time = time.time()
                sentenceBERT.fit(
                    training_data=train_df,
                    validation_data=valid_df,
                    batch_size=BATCH_SIZE,
                    num_epochs=NUM_EPOCHS,
                    device=DEVICE,
                    verbose=False,
                )
                elapsed_time = time.time() - start_time
                predictions: np.ndarray = sentenceBERT.predict(test_data=test_df, batch_size=BATCH_SIZE, device=DEVICE)
                del sentenceBERT
                torch.cuda.empty_cache()
                if "label" in test_df.columns:
                    final_labels: np.ndarray = test_df["label"].values

                    test_accuracy = accuracy_score(final_labels, predictions)
                    test_precision = precision_score(final_labels, predictions, average="weighted")
                    test_recall = recall_score(final_labels, predictions, average="weighted")
                    test_f1 = f1_score(final_labels, predictions, average="weighted")

                    acc.append(test_accuracy)
                    f1.append(test_f1)
                    precision.append(test_precision)
                    recall.append(test_recall)
                print(f"Iteration {i + 1} took {elapsed_time:.2f} seconds")

            print(f"\t{name} Average | {dataset_percentage * 100}% of original training data")
            print(
                f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | "
                f"P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
            )
