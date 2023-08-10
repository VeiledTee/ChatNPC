import json
from typing import List

import openai
import pandas as pd
import pinecone
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


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
        pinecone.init(
            api_key=api_keys[1],
            environment=api_keys[2],
        )

    df: pd.DataFrame = pd.read_csv("Data/contradiction-dataset.csv")
    HISTORY: List[dict] = []

    true: list = [1 if x.lower() == "contradiction" else 0 for x in df["gold_label"]]
    predictions: list = []
    for item in df.iterrows():
        sentence1: str = item[1]["sentence1"]
        sentence2: str = item[1]["sentence2"]
        label: int = 1 if item[1]["gold_label"].lower() == "contradiction" else 0
        print(f"{label}\t{sentence1}\t{sentence2}")
        predictions.append(
            int(
                answer(
                    f"Output 1 if following sentences contradictory, and 0 otherwise:\n{sentence1}\n{sentence2}",
                    HISTORY,
                )
            )
        )
        # print(answer(f'Output 1 if following sentences contradictory, and 0 otherwise:\n{sentence1}\n{sentence2}'))
