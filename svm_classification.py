import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from bilstm_training import load_txt_file_to_dataframe
import torch
from variables import DEVICE
from transformers import BertTokenizer, BertModel


CUSTOM_TEST_DATA: bool = False
BERT: bool = True


def encode_sentence(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


if BERT:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

    dataset_descriptors: list = ["match", "mismatch"]
    dataframes: list = []

    for descriptor in dataset_descriptors:
        df = pd.read_csv(f"Data/{descriptor}_cleaned.csv")
        dataframes.append(df)

    # Concatenate all the dataframes into a final dataframe
    multinli_df = pd.concat(dataframes, ignore_index=True)
    train_labels = multinli_df["label"].tolist()

    pair_x = [s.strip() for s in multinli_df["sentence1"]]
    pair_y = [s.strip() for s in multinli_df["sentence2"]]

    # Encode your sentence pairs using BERT
    train_embeddings = np.array([encode_sentence(f"{x} [SEP] {y}") for x, y in zip(pair_x, pair_y)])

    if CUSTOM_TEST_DATA:
        test_df = pd.read_csv("Data/contradiction-dataset.csv")
        test_labels = test_df["gold_label"].apply(lambda x: 1 if x.lower() == "contradiction" else 0).tolist()
        test_embeddings = np.array([encode_sentence(f"{x} [SEP] {y}") for x, y in zip(pair_x, pair_y)])
    else:
        test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned.csv")
        pair_x = [s.strip() for s in test_df["sentence1"]]
        pair_y = [s.strip() for s in test_df["sentence2"]]
        test_labels = test_df["label"].tolist()
        print(f"Percent Positive: {100 * sum(test_labels) / len(test_labels):.4f}")
        test_embeddings = np.array([encode_sentence(f"{x} [SEP] {y}") for x, y in zip(pair_x, pair_y)])
else:
    # Load the Sentence-BERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    dataset_descriptors: list = ["match", "mismatch"]
    dataframes: list = []

    for descriptor in dataset_descriptors:
        df = load_txt_file_to_dataframe(descriptor)
        dataframes.append(df)

    # Concatenate all the dataframes into a final dataframe
    multinli_df = pd.concat(dataframes, ignore_index=True)
    train_labels = [1 if y == "contradiction" else 0 for y in multinli_df["gold_label"]]

    pair_x = [s.strip() for s in multinli_df["sentence1"]]
    pair_y = [s.strip() for s in multinli_df["sentence2"]]

    sentences = [(x, y) for x, y in zip(pair_x, pair_y)]

    # Generate embeddings for sentence pairs
    train_embeddings = model.encode(sentences)

    if CUSTOM_TEST_DATA:
        test_df: pd.DataFrame = pd.read_csv("Data/contradiction-dataset.csv")
        pair_x: list = [s.strip() for s in test_df["sentence1"]]
        pair_y: list = [s.strip() for s in test_df["sentence2"]]

        sentences: list = [(x, y) for x, y in zip(pair_x, pair_y)]
        test_labels: list = [1 if y.lower() == "contradiction" else 0 for y in test_df["gold_label"]]

        # Generate embeddings for sentence pairs
        test_embeddings = model.encode(sentences)
        print(f"Custom test shape: {test_embeddings.shape}")
    else:
        test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned.csv")
        test_labels = test_df["label"].tolist()

        pair_x: list = [s.strip() for s in test_df["sentence1"]]
        pair_y: list = [s.strip() for s in test_df["sentence2"]]

        sentences: list = [(x, y) for x, y in zip(pair_x, pair_y)]
        # Generate embeddings for sentence pairs
        test_embeddings = model.encode(sentences)
        print(f"Custom test shape: {test_embeddings.shape}")

accuracies = []
f1_scores = []
for i in range(30):
    if CUSTOM_TEST_DATA:
        X_train = train_embeddings
        y_train = train_labels
        X_test = test_embeddings
        y_test = test_labels
    else:
        # Split the data into training and test sets, and get the indices
        X_train, X_test, y_train, y_test = train_test_split(train_embeddings, train_labels, test_size=0.2)

    # Train the SVM classifier
    svm_classifier = SVC(kernel="linear", C=float(15 / int(i + 1)))
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the SVM classifier
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\tIteration: {i} | C={15/(i+1)}")
    print("Accuracy:", accuracy)
    accuracies.append(accuracy)
    print("F1 score:", f1)
    f1_scores.append(f1)
print(f"\nAvg Acc: {sum(accuracies) / len(accuracies)}")
print(
    f"Best Acc: {max(accuracies)} | Index: {accuracies.index(max(accuracies))} | C: {15/(accuracies.index(max(accuracies)) + 1)}"
)
print(f"Avg F1:  {sum(f1_scores) / len(f1_scores)}")
print(
    f"Best F1: {max(f1_scores)} | Index: {f1_scores.index(max(f1_scores))} | C: {15/(f1_scores.index(max(f1_scores)) + 1)}"
)
