import json

import pinecone
from typing import List
import time
from chat import extract_name, get_information, load_file_information, upload, embed


def delete_all_vectors(names_of_characters) -> None:
    for i in range(len(names_of_characters)):
        character: str = list(names.keys())[i]
        data: str = f"Text Summaries/Summaries/{names_of_characters[character].lower()}.txt"
        namespace: str = extract_name(data).lower()
        index.delete(deleteAll=True, namespace=namespace)


if __name__ == "__main__":
    WINDOW: int = 3  # how many sentences are combined
    STRIDE: int = 2  # used to create overlap, num sentences we 'stride' over

    INDEX_NAME: str = "thesis-index"  # pinecone index name
    NAMESPACE = "peter-satoru"

    with open("keys.txt", "r") as key_file:
        api_keys = [key.strip() for key in key_file.readlines()]
        pinecone.init(
            api_key=api_keys[1],
            environment=api_keys[2],
        )

    index = pinecone.Index('thesis-index')

    # # upload data and generate query embedding.
    # embedded_query = embed("what fish can be found near ashbourne")
    #
    # # query Pinecone index and get context for model prompting.
    # responses = index.query(
    #     embedded_query,
    #     top_k=30,
    #     include_metadata=True,
    #     namespace=NAMESPACE,
    #     filter={
    #         "$or": [
    #             {"type": {"$eq": "background"}},
    #             {"type": {"$eq": "response"}},
    #         ]
    #     },
    # )
    #
    # print(len(responses['matches']))
    # print(responses)
    #
    # time.sleep(100)

    # Open file of characters and load its contents into a dictionary
    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    delete_all_vectors(names)
    print('deleted')

    # loop through characters and store background in database
    for i in range(len(names)):
        # CHARACTER: str = 'Melinda Deek'
        CHARACTER: str = list(names.keys())[i]
        PROFESSION, SOCIAL_CLASS = get_information(CHARACTER)
        print(CHARACTER, PROFESSION)

        DATA_FILE: str = f"Text Summaries/Summaries/{names[CHARACTER].lower()}.txt"

        NAMESPACE: str = extract_name(DATA_FILE).lower()
        char_info: List[str] = load_file_information(DATA_FILE)

        upload(
            NAMESPACE,
            load_file_information(DATA_FILE),
            index,
            "background",
        )

        print(f"{NAMESPACE} background uploaded")
