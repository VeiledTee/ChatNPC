import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from bilstm_training import load_txt_file_to_dataframe
import torch

from clean_dataset import create_subset_with_ratio, label_mapping
from variables import DEVICE
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt


def encode_sentence(language_model, tokenizer, sentence, device):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = language_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


if __name__ == '__main__':
    train_df = label_mapping(
        df=pd.read_csv("Data/MultiNLI/train_cleaned_subset.csv"),
        from_col='gold_label',
        to_col='label')
    valid_df = label_mapping(pd.read_csv("Data/MultiNLI/mismatch_cleaned.csv"))
    test_df = pd.read_csv("Data/MultiNLI/test_mismatch_cleaned.csv")

    for b in [True, False]:
        acc = []
        f1 = []
        precision = []
        recall = []
        for i in range(1):
            print(b, i)
            if b:
                bbu_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                bbu_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

                dataset_descriptors: list = ["match", "mismatch"]
                dataframes: list = []

                pair_x = [str(s).strip() for s in train_df["sentence1"]]
                pair_y = [str(s).strip() for s in train_df["sentence2"]]
                X_train = np.array(
                    [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in
                     zip(pair_x, pair_y)])
                y_train = train_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in valid_df["sentence1"]]
                pair_y = [str(s).strip() for s in valid_df["sentence2"]]
                X_val = np.array(
                    [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in
                     zip(pair_x, pair_y)])
                y_val = valid_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in test_df["sentence1"]]
                pair_y = [str(s).strip() for s in test_df["sentence2"]]
                X_test = np.array(
                    [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in
                     zip(pair_x, pair_y)])
                # y_test = test_df["gold_label"].tolist()
                # print(f"Percent Positive: {100 * sum([1 if int(i) == 2 else 0 for i in y_test]) / len(y_test):.4f}%")

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
                # # Calculate test set evaluation metrics
                # test_accuracy = accuracy_score(y_test, y_test_pred)
                # test_precision = precision_score(y_test, y_test_pred, average="weighted")
                # test_recall = recall_score(y_test, y_test_pred, average="weighted")
                # test_f1 = f1_score(y_test, y_test_pred, average="weighted")
                # # print(f'Test Accuracy: {test_accuracy:.2f}')
                # # print(f'Test Precision: {test_precision:.2f}')
                # # print(f'Test Recall: {test_recall:.2f}')
                # # print(f'Test F1-Score: {test_f1:.2f}')
                #
                # # Generate a classification report
                # class_report = classification_report(
                #     y_test, y_test_pred, target_names=["neutral", "entailment", "contradiction", '-']
                # )
                # # print("Classification Report:\n", class_report)
                #
                # acc.append(test_accuracy)
                # f1.append(test_f1)
                # precision.append(test_precision)
                # recall.append(test_recall)
                output_df: pd.DataFrame = pd.DataFrame({
                    'pairID': test_df['pairID'],
                    'gold_label': y_test_pred,
                })

                output_df = label_mapping(output_df, 'gold_label', 'gold_label')

                output_df.to_csv(f"Data/MultiNLI/SVM-BBU_mismatch.csv")
            else:
                # Load the Sentence-BERT model
                model = SentenceTransformer("all-MiniLM-L6-v2")

                dataset_descriptors: list = ["match", "mismatch"]
                dataframes: list = []

                pair_x = [str(s).strip() for s in train_df["sentence1"]]
                pair_y = [str(s).strip() for s in train_df["sentence2"]]
                X_train = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
                y_train = train_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in valid_df["sentence1"]]
                pair_y = [str(s).strip() for s in valid_df["sentence2"]]
                X_val = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
                y_val = valid_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in test_df["sentence1"]]
                pair_y = [str(s).strip() for s in test_df["sentence2"]]
                X_test = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
                # y_test = test_df["gold_label"].tolist()
                # print(f"Percent Positive: {100 * sum([1 if int(i) == 2 else 0 for i in y_test]) / len(y_test):.4f}%")

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
                # test_accuracy = accuracy_score(y_test, y_test_pred)
                # test_precision = precision_score(y_test, y_test_pred, average="weighted")
                # test_recall = recall_score(y_test, y_test_pred, average="weighted")
                # test_f1 = f1_score(y_test, y_test_pred, average="weighted")
                # # print(f'Test Accuracy: {test_accuracy:.2f}')
                # # print(f'Test Precision: {test_precision:.2f}')
                # # print(f'Test Recall: {test_recall:.2f}')
                # # print(f'Test F1-Score: {test_f1:.2f}')
                #
                # # Generate a classification report
                # class_report = classification_report(
                #     y_test, y_test_pred, target_names=["neutral", "entailment", "contradiction", '-']
                # )
                # # print("Classification Report:\n", class_report)
                # acc.append(test_accuracy)
                # f1.append(test_f1)
                # precision.append(test_precision)
                # recall.append(test_recall)

                output_df: pd.DataFrame = pd.DataFrame({
                    'pairID': test_df['pairID'],
                    'gold_label': y_test_pred,
                })

                output_df = label_mapping(output_df, 'gold_label', 'gold_label')

                output_df.to_csv(f"Data/MultiNLI/SVM-SBERT_mismatch.csv")

        # print(f"\t{b} Average")
        # print(
        #     f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
        # )
