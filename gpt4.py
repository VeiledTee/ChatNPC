import json
from typing import List

import openai
import pandas as pd
import pinecone
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report


def answer(prompt: str, chat_history: List[dict], is_chat: bool = True) -> str:
    """
    Using openAI API, respond to the provide prompt
    :param prompt: An engineered prompt to get the language model to respond to
    :param chat_history: the entire history of the conversation
    :param is_chat: are you chatting or looking for the completion of a phrase
    :return: The completed prompt
    """
    if is_chat:
        msgs: List[dict] = chat_history
        msgs.append({"role": "user", "content": prompt})  # build current history of conversation for model
        res: str = openai.ChatCompletion.create(
            model="gpt-4",
            messages=msgs,
            temperature=0,
        )  # conversation with LLM

        return res["choices"][0]["message"]["content"].strip()  # get model response
    else:
        res: str = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )  # LLM for phrase completion
        return res["choices"][0]["text"].strip()


if __name__ == "__main__":
    with open("keys.txt", "r") as key_file:
        api_keys = [key.strip() for key in key_file.readlines()]
        openai.api_key = api_keys[0]

    df: pd.DataFrame = pd.read_csv("Data/contradiction-dataset.csv").head(15)
    HISTORY: List[dict] = []

    true: list = [1 if x.lower() == "contradiction" else 0 for x in df["gold_label"]]
    print(f"Total: {sum(true)}")
    print(f"Positive records: {sum(true) / len(true)}")
    predictions: list = []
    false_neg: int = 0
    false_pos = 0
    for i, item in enumerate(df.iterrows()):
        sentence1: str = item[1]["sentence1"]
        sentence2: str = item[1]["sentence2"]
        label: int = true[i]
        # print(f"{label}\t{sentence1}\t{sentence2}")
        generated_answer = int(
                answer(
                    f"Output 1 if following sentences are contradictory, and 0 otherwise:\n{sentence1}\n{sentence2}",
                    HISTORY,
                )
            )
        predictions.append(generated_answer)

    accuracy = accuracy_score(true, predictions)
    f1 = f1_score(true, predictions)
    target_names = ['Negative', 'Positive']
    report = classification_report(true, predictions, target_names=target_names)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("\nClassification Report:")
    print(report)
"""
Total: 10
Positive records: 0.6666666666666666
Accuracy: 0.6
F1 Score: 0.7272727272727272

Classification Report:
              precision    recall  f1-score   support

    Negative       0.33      0.20      0.25         5
    Positive       0.67      0.80      0.73        10

    accuracy                           0.60        15
   macro avg       0.50      0.50      0.49        15
weighted avg       0.56      0.60      0.57        15
"""
