from typing import List

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ContradictDetectNN import embedding_to_tensor
from variables import DEVICE
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# # Load your data (assuming you have a DataFrame with 'sentence1', 'sentence2', and 'label' columns)
# train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
# valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
# test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")
#
# train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(embedding_to_tensor)
# train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(embedding_to_tensor)
# valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(embedding_to_tensor)
# valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(embedding_to_tensor)
# test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(embedding_to_tensor)
# test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(embedding_to_tensor)
#
# train_df["concatenated_embeddings"] = train_df.apply(
#     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
# )
# valid_df["concatenated_embeddings"] = valid_df.apply(
#     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
# )
# test_df["concatenated_embeddings"] = test_df.apply(
#     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
# )
#
# # Clean data
# X_train: torch.Tensor = torch.stack(list(train_df["concatenated_embeddings"]), dim=0)
# y_train: torch.Tensor = torch.tensor(train_df["label"].values, dtype=torch.long)
# X_valid: torch.Tensor = torch.stack(list(valid_df["concatenated_embeddings"]), dim=0)
# y_valid: torch.Tensor = torch.tensor(valid_df["label"].values, dtype=torch.long)
# X_test: torch.Tensor = torch.stack(list(test_df["concatenated_embeddings"]), dim=0)
# y_test: torch.Tensor = torch.tensor(test_df["label"].values, dtype=torch.long)
#
# # Define the parameter grid to search
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
#     'learning_rate': [0.01, .05, 0.1, 0.2, .5]
# }
#
# # Create an XGBoost classifier
# xgb_model = xgb.XGBClassifier()
#
# # Create a grid search object with cross-validation (e.g., 5-fold cross-validation)
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
#
# # Fit the grid search to your data
# grid_search.fit(X_train, y_train)
#
# # Get the best hyperparameters
# best_params = grid_search.best_params_
#
# print(best_params)
#
# # Train a model with the best hyperparameters
# best_xgb_model = xgb.XGBClassifier(**best_params)
# best_xgb_model.fit(X_train, y_train)
#
# # Evaluate the best model on your validation set (assuming you have X_valid and y_valid)
# validation_accuracy = best_xgb_model.score(X_valid, y_valid)
# print(validation_accuracy)

NUM_CLASSES: int = 3
acc: List[float] = []
f1: List[float] = []
precision: List[float] = []
recall: List[float] = []
for i in range(30):
    # Load your data (assuming you have a DataFrame with 'sentence1', 'sentence2', and 'label' columns)
    train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
    valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
    test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")

    train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(embedding_to_tensor)
    train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(embedding_to_tensor)
    valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(embedding_to_tensor)
    valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(embedding_to_tensor)
    test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(embedding_to_tensor)
    test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(embedding_to_tensor)

    train_df["concatenated_embeddings"] = train_df.apply(
        lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    )
    valid_df["concatenated_embeddings"] = valid_df.apply(
        lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    )
    test_df["concatenated_embeddings"] = test_df.apply(
        lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    )

    # Clean data
    X_train: torch.Tensor = torch.stack(list(train_df["concatenated_embeddings"]), dim=0)
    y_train: torch.Tensor = torch.tensor(train_df["label"].values, dtype=torch.long)
    X_valid: torch.Tensor = torch.stack(list(valid_df["concatenated_embeddings"]), dim=0)
    y_valid: torch.Tensor = torch.tensor(valid_df["label"].values, dtype=torch.long)
    X_test: torch.Tensor = torch.stack(list(test_df["concatenated_embeddings"]), dim=0)
    y_test: torch.Tensor = torch.tensor(test_df["label"].values, dtype=torch.long)

    # Train an XGBoost classifier
    model: xgb.XGBClassifier = xgb.XGBClassifier(objective="multi:softmax", num_class=3, n_jobs=-1, learning_rate=0.2, max_depth=4, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    predictions: np.ndarray = model.predict(X_test)

    unique_values, counts = np.unique(predictions, return_counts=True)
    # Print unique values and their counts
    for value, count in zip(unique_values, counts):
        print(f"Class {value}: {count} predictions")

    # evaluate
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision = precision_score(y_test, predictions, average="weighted")
    test_recall = recall_score(y_test, predictions, average="weighted")
    test_f1 = f1_score(y_test, predictions, average="weighted")

    acc.append(test_accuracy)
    f1.append(test_f1)
    precision.append(test_precision)
    recall.append(test_recall)
print(
    f"\tXGBoost Average\n{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
)
