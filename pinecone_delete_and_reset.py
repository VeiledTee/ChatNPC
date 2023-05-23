import json
import time
from typing import List

import pinecone
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def get_information(character_name):
    """
    Return information from the character's
    :param character_name:
    :return:
    """
    with open("Text Summaries/character_objects.json", "r") as f:
        data = json.load(f)

    for character in data["characters"]:
        if character["name"].strip() == character_name.strip():
            return character["profession"], character["social status"]

    return None  # character not found


def extract_name(file_name: str) -> str:
    # split the string into a list of parts, using "/" as the delimiter
    parts = file_name.split("/")
    # take the last part of the list (i.e. "john_pebble.txt")
    filename = parts[-1]
    # split the filename into a list of parts, using "_" as the delimiter
    name_parts = filename.split("_")
    # join the name parts together with a dash ("-"), and remove the ".txt" extension
    name = "-".join(name_parts)
    return name[:-4]


def load_file_information(load_file: str) -> List[str]:
    """
    Loads data from a file into a list of strings for embedding.
    :param load_file: Path to a file containing the data we want to encrypt.
    :return: A list of all the sentences in the load_file.
    """
    with open(load_file, "r") as text_file:
        # Use a list comprehension to clean and split the text file.
        char_string = [
            s.strip() + "."
            for line in text_file
            for s in line.strip().split(".")
            if s.strip()
        ]
    with open('Text Summaries/ashbourne.txt', 'r') as town_file:
        town_string = [
            s.strip() + "."
            for line in town_file
            for s in line.strip().split(".")
            if s.strip()
        ]
    char_string.extend(town_string)
    return char_string


def embed(query: str) -> List[float]:
    """
    Take a sentence of text and return the 384-dimension embedding
    :param query: The sentence to be embedded
    :return: Embedding representation of the sentence
    """
    # create SentenceTransformer model and embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model.encode(query).tolist()


def namespace_exist(namespace: str) -> bool:
    index = pinecone.Index("thesis-index")
    responses = index.query(
        embed("he is"),
        top_k=3,
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "answer"}},
            ]
        },
    )
    return responses["matches"] != []


def upload(
    namespace: str,
    data: List[str],
    text_type: str = "background",
    index_name: str = "thesis-index",
) -> None:
    """
    Upserts text embedding vectors into pinecone DB at the specific index
    :param data: Data to be embedded and stored
    :param text_type: The type of text we are embedding. Choose "background", "answer", or "question". Default value: "background"
    :param index_name: the name of the pinecone index to save the data to
    """
    if not pinecone.list_indexes():  # check if there are any indexes
        # create index if it doesn't exist
        pinecone.create_index(index_name, dimension=384)

    # if namespace_exist(namespace) and text_type == "background":
    #     return None

    # connect to pinecone and retrieve index
    index = pinecone.Index(INDEX_NAME)

    # create SentenceTransformer model and embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # results = index.query(embed(""), top_k=10000, namespace=NAMESPACE)
    # ids = [i["id"] for i in results["matches"]]

    # upload data to pinecone index
    for j in tqdm(range(0, len(data), STRIDE)):
        j_end = min(j + WINDOW, len(data))  # get end of batch
        ids = [str(x) for x in range(j, j_end)]  # generate ID
        metadata = [
            {"text": text, "type": text_type} for text in data[j:j_end]
        ]  # generate metadata
        embeddings = model.encode(data[j:j_end]).tolist()  # get embeddings
        curr_record = zip(
            ids, embeddings, metadata
        )  # compile into single vector
        index.upsert(vectors=curr_record, namespace=namespace)


if __name__ == "__main__":
    WINDOW: int = 3  # how many sentences are combined
    STRIDE: int = 2  # used to create overlap, num sentences we 'stride' over

    INDEX_NAME: str = "thesis-index"  # pinecone index name
    NAMESPACE = "melinda-deek"
    pinecone.init(

    )  # initialize pinecone env

    # index = pinecone.Index('thesis-index')
    #
    # # upload data and generate query embedding.
    # embedded_query = embed("what flowers can be found near ashbourne")
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
    # index.delete(delete_all=True, namespace="melinda-deek")

    pinecone.delete_index(INDEX_NAME)  # delete old index
    print(f"{INDEX_NAME} deleted")

    # Open file of characters and load its contents into a dictionary
    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    # loop through characters and store background in database
    for i in range(len(names)):
        # CHARACTER: str = 'Melinda Deek'
        CHARACTER: str = list(names.keys())[i]
        PROFESSION, SOCIAL_CLASS = get_information(CHARACTER)
        print(CHARACTER, PROFESSION)

        DATA_FILE: str = f"Text Summaries/{names[CHARACTER].lower()}.txt"

        NAMESPACE: str = extract_name(DATA_FILE).lower()

        upload(
            NAMESPACE,
            load_file_information(DATA_FILE),
            "background",
            INDEX_NAME,
        )

        print(f"{NAMESPACE} background uploaded")
