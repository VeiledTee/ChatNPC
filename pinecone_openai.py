import random
from typing import List
import openai
from sentence_transformers import SentenceTransformer
import torch
import pinecone
from tqdm.auto import tqdm
import time
import json


def extract_name(file_name: str) -> str:
    # split the string into a list of parts, using "/" as the delimiter
    parts = file_name.split("/")
    # take the last part of the list (i.e. "john_pebble.txt")
    filename = parts[-1]
    # split the filename into a list of parts, using "_" as the delimiter
    name_parts = filename.split("_")
    # join the name parts together with a dash ("-"), and remove the ".txt" extension
    name = "-".join(name_parts[:-1])
    return name


def get_profession(character_name):
    with open('Text Summaries/character_objects.json', 'r') as f:
        data = json.load(f)

    for character in data['characters']:
        if character['name'].strip() == character_name.strip():
            return character['profession']

    return None # character not found


def prompt_engineer(prompt: str, receiver: str, job: str, context: List[str]) -> str:
    """
    Given a base query and context, format it to be used as prompt
    :param job:
    :param receiver: The character being prompted
    :param prompt: The prompt query
    :param context: The context to be used in the prompt
    :return: The formatted prompt
    """
    prompt_start = (
        f"Answer the question as a {job}, based on the context below. "
        "Suggest a friend who may have an answer but only if you don't."
        "Be over the top.\n\n"
        + "Context: "
    )
    prompt_end = f"\n\nQuestion: {prompt}\nAnswer: "
    prompt_middle = ""
    # append contexts until hitting limit
    for c in context:
        prompt_middle += f"\n{c}"
    return prompt_start + prompt_middle + prompt_end


def answer(prompt: str) -> str:
    """
    Using openAI API, respond ot the provide prompt
    :param prompt: An engineered prompt to get the language model to respond to
    :return: The completed prompt
    """
    res: str = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    return res["choices"][0]["text"].strip()


def load(load_file: str) -> List[str]:
    """
    Loads data from a file into a list of strings for embedding.
    :param load_file: Path to a file containing the data we want to encrypt.
    :return: A list of all the sentences in the load_file.
    """
    with open(load_file, "r") as text_file:
        # Use a list comprehension to clean and split the text file.
        clean_string = [
            s.strip() + "."
            for line in text_file
            for s in line.strip().split(".")
            if s.strip()
        ]
    print("Data collected")
    return clean_string


def upload_generate(
    data: List[str], query: str, config: dict, index_name: str
) -> List[int]:
    if not pinecone.list_indexes():  # check if there are any indexes
        # create index if it doesn't exist
        pinecone.create_index(index_name, dimension=384, metadata_config=config)
        print(f"Index created: {index_name}")
    else:
        # print info about existing index
        index_info = pinecone.describe_index(INDEX_NAME)
        print(f"Index description: {index_info}")

    # connect ot pinecone and retrieve index
    index = pinecone.Index(INDEX_NAME)

    # wait for 30 seconds
    time.sleep(30)

    # create SentenceTransformer model and get query embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    embedded_query = model.encode(query).tolist()

    # upload data to pinecone index
    for i in tqdm(range(0, len(data), STRIDE)):
        i_end = min(i + WINDOW, len(data))  # get end of batch
        ids = [str(x) for x in range(i, i_end)]  # generate ID
        metadata = [{"text": text} for text in data[i:i_end]]  # generate metadata
        embeddings = model.encode(data[i:i_end]).tolist()  # get embeddings
        curr_record = zip(ids, embeddings, metadata)  # compile into single vector
        index.upsert(vectors=curr_record, namespace=NAMESPACE)
    print("Data Uploaded\nQuery Embedding Created")
    return embedded_query


def run_query_and_generate_answer(
    data: List[str],
    query: str,
    receiver: str,
    job: str,
    config: dict,
    index_name: str,
    save: bool = True,
) -> str:
    """
    Runs a query on a Pinecone index and generates an answer based on the response context.
    :param job:
    :param save:
    :param receiver: The character being prompted
    :param data: The data to be used for the query.
    :param query: The query to be run on the index.
    :param config: Configuration options for uploading the data to the index.
    :param index_name: The name of the Pinecone index.
    :return: The generated answer based on the response context.
    """
    # connect to index
    index = pinecone.Index(INDEX_NAME)

    # upload data and generate query embedding.
    embedded_query = upload_generate(
        data=data, query=query, config=config, index_name=index_name
    )

    # query Pinecone index and get context for model prompting.
    responses = index.query(
        embedded_query, top_k=5, include_metadata=True, namespace=NAMESPACE
    )
    context = [x["metadata"]["text"] for x in responses["matches"]]

    # generate clean prompt and answer.
    clean_prompt = prompt_engineer(query, receiver, job, context)
    generated_answer = answer(clean_prompt)

    if save:
        # save results to file and return generated answer.
        with open("prompt_response.txt", "a") as save_file:
            save_file.write("\n" + "=" * 120 + "\n")
            save_file.write(f"Prompt: {query}\n")
            save_file.write(f"To: {receiver}, a {job}\n")
            clean_prompt = clean_prompt.replace('\n', ' ')
            save_file.write(f"{clean_prompt}\n{generated_answer}")
        print("Results saved to file.")
    else:
        print(generated_answer)

    # pinecone.delete_index(index_name)

    return generated_answer

for i in range(7):
    WINDOW: int = 4  # how many sentences are combined
    STRIDE: int = 1  # used to create overlap, no. sentences we 'stride' over
    # Open the file and load its contents into a dictionary
    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    CHARACTER: str = list(names.keys())[i]
    # CHARACTER: str = random.choice(list(names.keys()))
    # CHARACTER: str = "Caleb Brown"
    PROFESSION: str = get_profession(CHARACTER)
    print(CHARACTER, PROFESSION)
    # print(names)
    DATA_FILE: str = f"Text Summaries/{names[CHARACTER]}.txt"

    INDEX_NAME: str = "thesis-index"
    NAMESPACE: str = extract_name(DATA_FILE)

    QUERY: str = "What dangers can be found around the village?"

    file_data = load(DATA_FILE)
    metadata_config = {"text": "query"}

    openai.api_key = ""  # windows 11
    pinecone.init(
    )

    run_query_and_generate_answer(
        data=file_data,
        receiver=CHARACTER,
        job=PROFESSION,
        query=QUERY,
        config=metadata_config,
        index_name=INDEX_NAME,
    )
