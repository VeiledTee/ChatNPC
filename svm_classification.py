import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from bilstm_training import load_txt_file_to_dataframe
import torch
from variables import DEVICE
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt


BERT: bool = True


def encode_sentence(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


for b in [True, False]:
    print(b)
    if b:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

        dataset_descriptors: list = ["match", "mismatch"]
        dataframes: list = []

        train_df = pd.read_csv("train.csv")
        valid_df = pd.read_csv("valid.csv")
        test_df = pd.read_csv("test.csv")

        pair_x = [s.strip() for s in train_df["sentence1"]]
        pair_y = [s.strip() for s in train_df["sentence2"]]
        X_train = np.array([encode_sentence(f"{x} [SEP] {y}") for x, y in zip(pair_x, pair_y)])
        y_train = train_df["gold_label"].tolist()
        pair_x = [s.strip() for s in valid_df["sentence1"]]
        pair_y = [s.strip() for s in valid_df["sentence2"]]
        X_val = np.array([encode_sentence(f"{x} [SEP] {y}") for x, y in zip(pair_x, pair_y)])
        y_val = valid_df["gold_label"].tolist()
        pair_x = [s.strip() for s in test_df["sentence1"]]
        pair_y = [s.strip() for s in test_df["sentence2"]]
        X_test = np.array([encode_sentence(f"{x} [SEP] {y}") for x, y in zip(pair_x, pair_y)])
        y_test = test_df["gold_label"].tolist()
        print(f"Percent Positive: {100 * sum([1 if int(i) == 2 else 0 for i in y_test]) / len(y_test):.4f}%")

        # Define the hyperparameter grid for C values
        param_grid = {"C": [0.1, 1, 10, 100]}

        # Initialize lists to store results
        validation_accuracies = []
        validation_precisions = []
        validation_recalls = []
        validation_f1_scores = []
        test_accuracies = []

        final_clf = SVC(kernel="linear", C=0.1)

        # Train the final model on the entire training dataset
        final_clf.fit(X_train, y_train)
        y_val_pred = final_clf.predict(X_val)
        # Calculate evaluation metrics
        validation_accuracy = accuracy_score(y_val, y_val_pred)
        validation_precision = precision_score(y_val, y_val_pred, average="weighted")
        validation_recall = recall_score(y_val, y_val_pred, average="weighted")
        validation_f1 = f1_score(y_val, y_val_pred, average="weighted")
        # print(f'Validation Accuracy: {validation_accuracy:.2f}')
        # print(f'Validation Precision: {validation_precision:.2f}')
        # print(f'Validation Recall: {validation_recall:.2f}')
        # print(f'Validation F1-Score: {validation_f1:.2f}')

        # Evaluate the final model on the test set (unseen data)
        y_test_pred = final_clf.predict(X_test)
        # Count unique values and their counts
        unique_values, counts = np.unique(y_test_pred, return_counts=True)
        # Print unique values and their counts
        for value, count in zip(unique_values, counts):
            print(f"Class {value}: {count} predictions")
        # Calculate test set evaluation metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average="weighted")
        test_recall = recall_score(y_test, y_test_pred, average="weighted")
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")
        # print(f'Test Accuracy: {test_accuracy:.2f}')
        # print(f'Test Precision: {test_precision:.2f}')
        # print(f'Test Recall: {test_recall:.2f}')
        # print(f'Test F1-Score: {test_f1:.2f}')

        # Generate a classification report
        class_report = classification_report(
            y_test, y_test_pred, target_names=["neutral", "entailment", "contradiction"]
        )
        print("Classification Report:\n", class_report)

    else:
        # Load the Sentence-BERT model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        dataset_descriptors: list = ["match", "mismatch"]
        dataframes: list = []

        train_df = pd.read_csv("train.csv")
        valid_df = pd.read_csv("valid.csv")
        test_df = pd.read_csv("test.csv")

        pair_x = [s.strip() for s in train_df["sentence1"]]
        pair_y = [s.strip() for s in train_df["sentence2"]]
        X_train = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
        y_train = train_df["gold_label"].tolist()
        pair_x = [s.strip() for s in valid_df["sentence1"]]
        pair_y = [s.strip() for s in valid_df["sentence2"]]
        X_val = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
        y_val = valid_df["gold_label"].tolist()
        pair_x = [s.strip() for s in test_df["sentence1"]]
        pair_y = [s.strip() for s in test_df["sentence2"]]
        X_test = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
        y_test = test_df["gold_label"].tolist()
        print(f"Percent Positive: {100 * sum([1 if int(i) == 2 else 0 for i in y_test]) / len(y_test):.4f}%")

        # Create the final SVM classifier with the best hyperparameters
        final_clf = SVC(kernel="linear", C=1)

        # Train the final model on the entire training dataset
        final_clf.fit(X_train, y_train)
        y_val_pred = final_clf.predict(X_val)
        # Calculate evaluation metrics
        validation_accuracy = accuracy_score(y_val, y_val_pred)
        validation_precision = precision_score(y_val, y_val_pred, average="weighted")
        validation_recall = recall_score(y_val, y_val_pred, average="weighted")
        validation_f1 = f1_score(y_val, y_val_pred, average="weighted")
        # print(f'Validation Accuracy: {validation_accuracy:.2f}')
        # print(f'Validation Precision: {validation_precision:.2f}')
        # print(f'Validation Recall: {validation_recall:.2f}')
        # print(f'Validation F1-Score: {validation_f1:.2f}')

        # Evaluate the final model on the test set (unseen data)
        y_test_pred = final_clf.predict(X_test)
        # Count unique values and their counts
        unique_values, counts = np.unique(y_test_pred, return_counts=True)
        # Print unique values and their counts
        for value, count in zip(unique_values, counts):
            print(f"Class {value}: {count} predictions")
        # Calculate test set evaluation metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average="weighted")
        test_recall = recall_score(y_test, y_test_pred, average="weighted")
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")
        # print(f'Test Accuracy: {test_accuracy:.2f}')
        # print(f'Test Precision: {test_precision:.2f}')
        # print(f'Test Recall: {test_recall:.2f}')
        # print(f'Test F1-Score: {test_f1:.2f}')

        # Generate a classification report
        class_report = classification_report(
            y_test, y_test_pred, target_names=["neutral", "entailment", "contradiction"]
        )
        print("Classification Report:\n", class_report)
# # Define a range of C values to test
# C_range = np.logspace(-10, 10, 21)
#
# # Calculate training and validation scores at different C values
# train_scores, valid_scores = validation_curve(
#     SVC(), train_embeddings, train_labels, param_name="C", param_range=C_range, cv=10
# )
#
# # Plot the validation curve
# plt.figure(figsize=(10, 6))
# plt.semilogx(C_range, np.mean(train_scores, axis=1), label="Training score", marker="o")
# plt.semilogx(C_range, np.mean(valid_scores, axis=1), label="Validation score", marker="o")
# plt.xlabel("C")
# plt.ylabel("Score")
# plt.legend()
# plt.grid()
# plt.savefig(f"Figures/{b}_validation_curve.svg", format="svg")

accuracies = []
f1_scores = []
# for i in range(30):
#     if CUSTOM_TEST_DATA:
#         X_train = train_embeddings
#         y_train = train_labels
#         X_test = test_embeddings
#         y_test = test_labels
#     else:
#         # Split the data into training and test sets, and get the indices
#         X_train, X_test, y_train, y_test = train_test_split(train_embeddings, train_labels, test_size=0.2)
#
#     # Train the SVM classifier
#     svm_classifier = SVC(kernel="linear", C=float(15 / int(i + 1)))
#     svm_classifier.fit(X_train, y_train)
#
#     # Make predictions on the test set
#     y_pred = svm_classifier.predict(X_test)
#
#     # Evaluate the SVM classifier
#     accuracy = accuracy_score(y_test, y_pred)
#     # report = classification_report(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     print(f"\tIteration: {i} | C={15/(i+1)}")
#     print("Accuracy:", accuracy)
#     accuracies.append(accuracy)
#     print("F1 score:", f1)
#     f1_scores.append(f1)
# print(f"\nAvg Acc: {sum(accuracies) / len(accuracies)}")
# print(
#     f"Best Acc: {max(accuracies)} | Index: {accuracies.index(max(accuracies))} | C: {15/(accuracies.index(max(accuracies)) + 1)}"
# )
# print(f"Avg F1:  {sum(f1_scores) / len(f1_scores)}")
# print(
#     f"Best F1: {max(f1_scores)} | Index: {f1_scores.index(max(f1_scores))} | C: {15/(f1_scores.index(max(f1_scores)) + 1)}"
# )
