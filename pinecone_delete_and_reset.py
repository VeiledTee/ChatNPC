import json

import pinecone
from typing import List
import time
from webchat import extract_name, get_information, load_file_information, upload_background
from global_functions import name_conversion


def delete_all_vectors(names_of_characters) -> None:
    for i in range(len(names_of_characters)):
        character: str = list(names.keys())[i]
        data: str = f"Text Summaries/Summaries/{names_of_characters[character].lower()}.txt"
        namespace: str = extract_name(data).lower()
        index.delete(deleteAll=True, namespace=namespace)


if __name__ == "__main__":
    with open("keys.txt", "r") as key_file:
        api_keys = [key.strip() for key in key_file.readlines()]
        pinecone.init(
            api_key=api_keys[1],
            environment=api_keys[2],
        )

    index = pinecone.Index("chatnpc-index")

    # Open file of characters and load its contents into a dictionary
    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    delete_all_vectors(names)
    print("deleted")

    # loop through characters and store background in database
    for i in range(len(names)):
        # CHARACTER: str = 'Melinda Deek'
        CHARACTER: str = list(names.keys())[i]
        PROFESSION, SOCIAL_CLASS = get_information(CHARACTER)
        print(CHARACTER, PROFESSION)

        DATA_FILE: str = f"Text Summaries/Summaries/{names[CHARACTER].lower()}.txt"

        NAMESPACE: str = name_conversion(to_snake=True, to_convert=CHARACTER)
        char_info: List[str] = load_file_information(DATA_FILE)

        upload_background(CHARACTER)

        print(f"{NAMESPACE} background uploaded for {CHARACTER}")
