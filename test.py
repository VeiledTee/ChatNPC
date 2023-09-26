from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from transformers import BertForSequenceClassification, BertTokenizer, AutoModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from ContradictDetectNN import count_negations, ph_to_tensor, embedding_to_tensor

# from variables import DEVICE
DEVICE = 'cpu'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from clean_dataset import create_subset_with_ratio
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


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


if __name__ == "__main__":
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 32
    NUM_CLASSES: int = 3

    acc = []
    f1 = []
    precision = []
    recall = []

    train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
    valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
    test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")

    # train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(embedding_to_tensor)
    # train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(embedding_to_tensor)
    # valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(embedding_to_tensor)
    # valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(embedding_to_tensor)
    # test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(embedding_to_tensor)
    # test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(embedding_to_tensor)
    #
    # sentence1_training_embeddings: torch.Tensor = torch.stack(list(train_df["sentence1_embeddings"]), dim=0).to(DEVICE)
    # sentence2_training_embeddings: torch.Tensor = torch.stack(list(train_df["sentence2_embeddings"]), dim=0).to(DEVICE)
    # sentence1_validation_embeddings: torch.Tensor = torch.stack(
    #     list(valid_df["sentence1_embeddings"]), dim=0
    # ).to(DEVICE)
    # sentence2_validation_embeddings: torch.Tensor = torch.stack(
    #     list(valid_df["sentence2_embeddings"]), dim=0
    # ).to(DEVICE)
    # sentence1_testing_embeddings: torch.Tensor = torch.stack(list(test_df["sentence1_embeddings"]), dim=0).to(DEVICE)
    # sentence2_testing_embeddings: torch.Tensor = torch.stack(list(test_df["sentence2_embeddings"]), dim=0).to(DEVICE)

    sentence1_training_embeddings, sentence2_training_embeddings = get_embeddings(train_df, 'roberta-base')
    sentence1_validation_embeddings, sentence2_validation_embeddings = get_embeddings(valid_df, 'roberta-base')
    sentence1_testing_embeddings, sentence2_testing_embeddings = get_embeddings(test_df, 'roberta-base')

    for column in ["sentence1_ph_a", "sentence1_ph_b", "sentence2_ph_a", "sentence2_ph_b"]:
        # Training cleaning
        train_df[column] = train_df[column].apply(ph_to_tensor)
        train_df[column] = train_df[column].apply(lambda x: x[:, -1])
        # Validation cleaning
        valid_df[column] = valid_df[column].apply(ph_to_tensor)
        valid_df[column] = valid_df[column].apply(lambda x: x[:, -1])
        # Testing cleaning
        test_df[column] = test_df[column].apply(ph_to_tensor)
        test_df[column] = test_df[column].apply(lambda x: x[:, -1])

    scaler = StandardScaler()

    train_s1_a, train_s1_b, train_s2_a, train_s2_b = get_features(train_df)
    valid_s1_a, valid_s1_b, valid_s2_a, valid_s2_b = get_features(valid_df)
    test_s1_a, test_s1_b, test_s2_a, test_s2_b = get_features(test_df)

    print("PH Formatted")

    training_input = np.concatenate(
        [sentence1_training_embeddings.cpu().numpy(),
         sentence2_training_embeddings.cpu().numpy(),
         train_s1_a,
         train_s1_b,
         train_s2_a,
         train_s2_b],
        axis=1,
    )
    print("Training Data")
    validation_input = np.concatenate(
        [sentence1_validation_embeddings.cpu().numpy(),
         sentence2_validation_embeddings.cpu().numpy(),
         valid_s1_a,
         valid_s1_b,
         valid_s2_a,
         valid_s2_b],
        axis=1,
    )
    print("Validation Data")
    testing_input = np.concatenate(
        [sentence1_testing_embeddings.cpu().numpy(),
         sentence2_testing_embeddings.cpu().numpy(),
         test_s1_a,
         test_s1_b,
         test_s2_a,
         test_s2_b],
        axis=1,
    )
    print("Testing Data")

    acc = []
    f1 = []
    precision = []
    recall = []
    for i in range(30):

        # Initialize SVM classifier
        svm_classifier = SVC(kernel='linear', C=1.0)

        # Hyperparameter tuning loop
        best_accuracy = 0
        best_svm = None

        for C_value in [0.01]:  # Example values for the regularization parameter C
            print(C_value)
            # Train SVM on the training set
            svm_classifier.set_params(C=0.01)
            svm_classifier.fit(training_input, train_df['label'].values.tolist())

            # Evaluate on the validation set
            y_val_pred = svm_classifier.predict(validation_input)
            accuracy = accuracy_score(valid_df['label'].values.tolist(), y_val_pred)

            print(f"C = {C_value}, Validation Accuracy = {accuracy:.4f}")

            # Check if this model is the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_svm = svm_classifier

        # Train the best SVM on the combined training and validation sets
        best_svm.fit(validation_input, valid_df['label'].values.tolist())

        # Evaluate on the test set
        y_test_pred = best_svm.predict(testing_input)

        test_accuracy = accuracy_score(test_df['label'].values.tolist(), y_test_pred)
        test_f1 = f1_score(test_df['label'].values.tolist(), y_test_pred, average="weighted")
        test_precision = precision_score(test_df['label'].values.tolist(), y_test_pred, average="weighted")
        test_recall = recall_score(test_df['label'].values.tolist(), y_test_pred, average="weighted")

        acc.append(test_accuracy)
        f1.append(test_f1)
        precision.append(test_precision)
        recall.append(test_recall)

    print(f"\tAverage")
    print(
        f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | "
        f"P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
    )
