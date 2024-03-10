from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ContradictDetectNN import embedding_to_tensor
from clean_dataset import encode_sentence, label_mapping

from config import DEVICE

# DEVICE = 'cpu'

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb


def x_and_y_from_data(df: pd.DataFrame, encoder, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    pair_x = [str(s).strip() for s in df["sentence1"]]  # Isolate sentence 1
    pair_y = [str(s).strip() for s in df["sentence2"]]  # Isolate sentence 2
    X = np.array(
        [encode_sentence(encoder, tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in zip(pair_x, pair_y)]
    )  # retrieve all sentence embeddings for each pair of sentences
    if "label" in df.columns:
        y = np.array([int(value) for value in df["label"].tolist()])
    else:
        y = np.array([])
    return X, y


if __name__ == "__main__":
    for dataset in ["match", "mismatch"]:
        training_file: str = "Data/MultiNLI/train_cleaned_subset.csv"
        validation_file: str = f"Data/MultiNLI/{dataset}_cleaned.csv"
        testing_file: str = f"Data/MultiNLI/test_{dataset}_cleaned.csv"

        # Load the BERT model and tokenizer
        model_name = "bert-base-uncased"
        bbu_tokenizer = BertTokenizer.from_pretrained(model_name)
        bbu_model = BertModel.from_pretrained(model_name).to(DEVICE)

        NUM_CLASSES: int = 3
        acc: List[float] = []
        f1: List[float] = []
        precision: List[float] = []
        recall: List[float] = []

        # Load your data (assuming you have a DataFrame with 'sentence1', 'sentence2', and 'label' columns)
        train_df: pd.DataFrame = pd.read_csv(training_file)
        valid_df: pd.DataFrame = pd.read_csv(validation_file)
        test_df: pd.DataFrame = pd.read_csv(testing_file)

        X_train, y_train = x_and_y_from_data(df=train_df, encoder=bbu_model, tokenizer=bbu_tokenizer)
        print(X_train.shape, y_train.shape)
        print("Training loaded")
        X_val, y_val = x_and_y_from_data(df=valid_df, encoder=bbu_model, tokenizer=bbu_tokenizer)
        print(X_val.shape, y_val.shape)
        print("Validation loaded")
        X_test, y_test = x_and_y_from_data(df=test_df, encoder=bbu_model, tokenizer=bbu_tokenizer)
        print(X_test.shape)
        print("Testing loaded")

        # # Create an XGBoost classifier
        # xgb_model: xgb.XGBClassifier = xgb.XGBClassifier()
        #
        # param_grid: dict = {
        #     'n_estimators': [100, 200, 300, 400, 500],
        #     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        #     'learning_rate': [0.01, .05, 0.1, 0.2, 0.5]
        # }
        #
        # # Create a grid search object with cross-validation (e.g., 5-fold cross-validation)
        # grid_search: GridSearchCV = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5,
        #                                          n_jobs=-1)
        #
        # # Fit the grid search to your data
        # grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 300}

        # print(best_params, type(best_params))
        #
        # # Train a model with the best hyperparameters
        # best_xgb_model: xgb.XGBClassifier = xgb.XGBClassifier(**best_params)
        # best_xgb_model.fit(X_train, y_train)
        #
        # # Evaluate the best model on your validation set (assuming you have X_val and y_val)
        # validation_accuracy = best_xgb_model.score(X_val, y_val)
        # print(f"Validation accuracy: {validation_accuracy}")

        # Train an XGBoost classifier
        model: xgb.XGBClassifier = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=NUM_CLASSES,
            n_jobs=-1,
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            n_estimators=best_params["n_estimators"],
        )
        # match {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 300}
        # mismatch {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 300}
        # Fit classifier
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # fit model
        # Make predictions
        predictions: np.ndarray = model.predict(X_test)

        if training_file.split("/")[1] == "MultiNLI":
            # Save prediction for kaggle submission
            output_df: pd.DataFrame = pd.DataFrame(
                {
                    "pairID": test_df["pairID"],
                    "gold_label": predictions,
                }
            )

            output_df = label_mapping(output_df, "gold_label", "gold_label", str_to_int=False)

            output_df.to_csv(f"Data/MultiNLI/XGBoost_{dataset}.csv", index=False)
            print(f"{dataset} saved")
        else:
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
                f"\tXGBoost Average\n{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} "
                f"| P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
            )
