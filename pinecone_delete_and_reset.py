import json

import pinecone

from chat import extract_name, get_information, load_file_information, upload

if __name__ == "__main__":
    WINDOW: int = 3  # how many sentences are combined
    STRIDE: int = 2  # used to create overlap, num sentences we 'stride' over

    INDEX_NAME: str = "thesis-index"  # pinecone index name
    NAMESPACE = "melinda-deek"

    with open("keys.txt", "r") as key_file:  # initialize
        pinecone.init(
            api_key=key_file.readlines()[1],
            environment=key_file.readlines()[2],
        )

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
