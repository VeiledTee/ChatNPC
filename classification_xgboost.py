from typing import List

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ContradictDetectNN import embedding_to_tensor
from classification_svm import encode_sentence
from clean_dataset import label_mapping

# from variables import DEVICE
DEVICE = 'cpu'

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


def objective(trial):
    train_df = pd.read_csv("Data/MultiNLI/train_cleaned_subset.csv")
    valid_df = pd.read_csv("Data/MultiNLI/match_cleaned.csv")

    model_name = 'bert-base-uncased'
    bbu_tokenizer = BertTokenizer.from_pretrained(model_name)
    bbu_model = BertModel.from_pretrained(model_name).to(DEVICE)

    pair_x = [s.strip() for s in train_df["sentence1"]]
    pair_y = [s.strip() for s in train_df["sentence2"]]
    train_x = np.array(
        [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in zip(pair_x, pair_y)])
    train_y = train_df["gold_label"].tolist()
    pair_x = [s.strip() for s in valid_df["sentence1"]]
    pair_y = [s.strip() for s in valid_df["sentence2"]]
    valid_x = np.array(
        [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in zip(pair_x, pair_y)])
    valid_y = valid_df["gold_label"].tolist()

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == '__main__':
    # # Load the BERT model and tokenizer
    # model_name = 'bert-base-uncased'
    # bbu_tokenizer = BertTokenizer.from_pretrained(model_name)
    # bbu_model = BertModel.from_pretrained(model_name).to(DEVICE)
    #
    # # # Load your data (assuming you have a DataFrame with 'sentence1', 'sentence2', and 'label' columns)
    # # train_df = pd.read_csv("Data/SemEval2014T1/train_cleaned_ph.csv")
    # # valid_df = pd.read_csv("Data/SemEval2014T1/valid_cleaned_ph.csv")
    # # test_df = pd.read_csv("Data/SemEval2014T1/test_cleaned_ph.csv")
    # #
    # # train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(embedding_to_tensor)
    # # train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(embedding_to_tensor)
    # # valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(embedding_to_tensor)
    # # valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(embedding_to_tensor)
    # # test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(embedding_to_tensor)
    # # test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(embedding_to_tensor)
    # #
    # # train_df["concatenated_embeddings"] = train_df.apply(
    # #     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    # # )
    # # valid_df["concatenated_embeddings"] = valid_df.apply(
    # #     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    # # )
    # # test_df["concatenated_embeddings"] = test_df.apply(
    # #     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    # # )
    # #
    # # # Clean data
    # # X_train: torch.Tensor = torch.stack(list(train_df["concatenated_embeddings"]), dim=0)
    # # y_train: torch.Tensor = torch.tensor(train_df["label"].values, dtype=torch.long)
    # # X_val: torch.Tensor = torch.stack(list(valid_df["concatenated_embeddings"]), dim=0)
    # # y_val: torch.Tensor = torch.tensor(valid_df["label"].values, dtype=torch.long)
    # # X_test: torch.Tensor = torch.stack(list(test_df["concatenated_embeddings"]), dim=0)
    # # y_test: torch.Tensor = torch.tensor(test_df["label"].values, dtype=torch.long)
    # #
    # # # Define the parameter grid to search
    # # param_grid = {
    # #     'n_estimators': [100, 200, 300, 400, 500],
    # #     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    # #     'learning_rate': [0.01, .05, 0.1, 0.2, .5]
    # # }
    # #
    # # # Create an XGBoost classifier
    # # xgb_model = xgb.XGBClassifier()
    # #
    # # # Create a grid search object with cross-validation (e.g., 5-fold cross-validation)
    # # grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
    # #
    # # # Fit the grid search to your data
    # # grid_search.fit(X_train, y_train)
    # #
    # # # Get the best hyperparameters
    # # best_params = grid_search.best_params_
    # #
    # # print(best_params)
    # #
    # # # Train a model with the best hyperparameters
    # # best_xgb_model = xgb.XGBClassifier(**best_params)
    # # best_xgb_model.fit(X_train, y_train)
    # #
    # # # Evaluate the best model on your validation set (assuming you have X_val and y_val)
    # # validation_accuracy = best_xgb_model.score(X_val, y_val)
    # # print(validation_accuracy)
    #
    # NUM_CLASSES: int = 3
    # acc: List[float] = []
    # f1: List[float] = []
    # precision: List[float] = []
    # recall: List[float] = []
    # for i in range(1):
    #     # Load your data (assuming you have a DataFrame with 'sentence1', 'sentence2', and 'label' columns)
    #     train_df = pd.read_csv("Data/MultiNLI/train_cleaned_subset.csv")
    #     valid_df = pd.read_csv("Data/MultiNLI/match_cleaned.csv")
    #     test_df = pd.read_csv("Data/MultiNLI/test_match_cleaned.csv")
    #
    #     # train_df["sentence1_embeddings"] = train_df["sentence1_embeddings"].apply(embedding_to_tensor)
    #     # train_df["sentence2_embeddings"] = train_df["sentence2_embeddings"].apply(embedding_to_tensor)
    #     # valid_df["sentence1_embeddings"] = valid_df["sentence1_embeddings"].apply(embedding_to_tensor)
    #     # valid_df["sentence2_embeddings"] = valid_df["sentence2_embeddings"].apply(embedding_to_tensor)
    #     # test_df["sentence1_embeddings"] = test_df["sentence1_embeddings"].apply(embedding_to_tensor)
    #     # test_df["sentence2_embeddings"] = test_df["sentence2_embeddings"].apply(embedding_to_tensor)
    #
    # pair_x = [s.strip() for s in train_df["sentence1"]]
    # pair_y = [s.strip() for s in train_df["sentence2"]]
    # X_train = np.array(
    #     [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in zip(pair_x, pair_y)])
    # y_train = train_df["gold_label"].tolist()
    # pair_x = [s.strip() for s in valid_df["sentence1"]]
    # pair_y = [s.strip() for s in valid_df["sentence2"]]
    # X_val = np.array(
    #     [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in zip(pair_x, pair_y)])
    # y_val = valid_df["gold_label"].tolist()
    # pair_x = [s.strip() for s in test_df["sentence1"]]
    # pair_y = [s.strip() for s in test_df["sentence2"]]
    # X_test = np.array(
    #     [encode_sentence(bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE) for x, y in zip(pair_x, pair_y)])
    # y_test = valid_df["gold_label"].tolist()
    #
    #     # train_df["concatenated_embeddings"] = train_df.apply(
    #     #     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    #     # )
    #     # valid_df["concatenated_embeddings"] = valid_df.apply(
    #     #     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    #     # )
    #     # test_df["concatenated_embeddings"] = test_df.apply(
    #     #     lambda row: torch.cat([row["sentence1_embeddings"], row["sentence2_embeddings"]]), axis=1
    #     # )
    #     #
    #     # # Clean data
    #     # X_train: torch.Tensor = torch.stack(list(train_df["concatenated_embeddings"]), dim=0)
    #     # y_train: torch.Tensor = torch.tensor(train_df["label"].values, dtype=torch.long)
    #     # X_val: torch.Tensor = torch.stack(list(valid_df["concatenated_embeddings"]), dim=0)
    #     # y_val: torch.Tensor = torch.tensor(valid_df["label"].values, dtype=torch.long)
    #     # X_test: torch.Tensor = torch.stack(list(valid_df["concatenated_embeddings"]), dim=0)
    #     # y_test: torch.Tensor = torch.tensor(valid_df["label"].values, dtype=torch.long)
    #
    #     # Create an XGBoost classifier
    #     xgb_model = xgb.XGBClassifier()
    #
    #     param_grid = {
    #         'n_estimators': [100, 200, 300, 400, 500],
    #         'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'learning_rate': [0.01, .05, 0.1, 0.2, .5]
    #     }
    #
    #     # Create a grid search object with cross-validation (e.g., 5-fold cross-validation)
    #     grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
    #
    #     # Fit the grid search to your data
    #     grid_search.fit(X_train, y_train)
    #
    #     # Get the best hyperparameters
    #     best_params = grid_search.best_params_
    #
    #     print(best_params)
    #
    #     # Train a model with the best hyperparameters
    #     best_xgb_model = xgb.XGBClassifier(**best_params)
    #     best_xgb_model.fit(X_train, y_train)
    #
    #     # Evaluate the best model on your validation set (assuming you have X_val and y_val)
    #     validation_accuracy = best_xgb_model.score(X_val, y_val)
    #     print(validation_accuracy)
    #
    # #     # Train an XGBoost classifier
    # #     model: xgb.XGBClassifier = xgb.XGBClassifier(objective="multi:softmax", num_class=3, n_jobs=-1,
    # #                                                  learning_rate=0.2,
    # #                                                  max_depth=4, n_estimators=500)
    # #     model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    # #     predictions: np.ndarray = model.predict(X_test)
    # #     # output_df: pd.DataFrame = pd.DataFrame({
    # #     #     'pairID': test_df['pairID'],
    # #     #     'gold_label': predictions,
    # #     # })
    # #     #
    # #     # output_df = label_mapping(output_df, 'gold_label', 'gold_label')
    # #     #
    # #     # output_df.to_csv(f"Data/MultiNLI/XGBoost_matched.csv")
    # #
    # #     unique_values, counts = np.unique(predictions, return_counts=True)
    # #     # Print unique values and their counts
    # #     for value, count in zip(unique_values, counts):
    # #         print(f"Class {value}: {count} predictions")
    # #
    # #     # evaluate
    # #     test_accuracy = accuracy_score(y_test, predictions)
    # #     test_precision = precision_score(y_test, predictions, average="weighted")
    # #     test_recall = recall_score(y_test, predictions, average="weighted")
    # #     test_f1 = f1_score(y_test, predictions, average="weighted")
    # #
    # #     acc.append(test_accuracy)
    # #     f1.append(test_f1)
    # #     precision.append(test_precision)
    # #     recall.append(test_recall)
    # # print(
    # #     f"\tXGBoost Average\n{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
    # # )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600, n_jobs=-1)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"\tValue: {trial.value}")
    print("\tParams:")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")
