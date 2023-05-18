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
    name = "-".join(name_parts)
    return name[:-4]


def namespace_exist(namespace: str) -> bool:
    index = pinecone.Index("thesis-index")
    responses = index.query(
        embed("he is"), top_k=3, include_metadata=True, namespace=namespace, filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "answer"}}
            ]
        }
    )
    return responses['matches'] != []


def get_information(character_name):
    """
    Return information from the character's
    :param character_name:
    :return:
    """
    with open('Text Summaries/character_objects.json', 'r') as f:
        data = json.load(f)

    for character in data['characters']:
        if character['name'].strip() == character_name.strip():
            return character['profession'], character['social status']

    return None  # character not found


def prompt_engineer(prompt: str, receiver: str, job: str, status: str, context: List[str]) -> str:
    """
    Given a base query and context, format it to be used as prompt
    :param job:
    :param receiver: The character being prompted
    :param prompt: The prompt query
    :param context: The context to be used in the prompt
    :return: The formatted prompt
    """
    prompt_start = (
            f"Reply as {receiver}, a {job}, based on the following context. "
            f"Only reference those explicitly mentioned in the following context if necessary. "
            f"One sentence. Use {GRAMMAR[status.split()[0]]} grammar.\nContext:"
    )
    prompt_end = f"\n\nQuestion: {prompt}\nAnswer: "
    prompt_middle = ""
    # append contexts until hitting limit
    for c in context:
        prompt_middle += f"\n{c}"
    return prompt_start + prompt_middle + prompt_end


def answer(prompt: str, character: str) -> str:
    """
    Using openAI API, respond ot the provide prompt
    :param prompt: An engineered prompt to get the language model to respond to
    :return: The completed prompt
    """
    # res: str = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     temperature=0,
    #     max_tokens=400,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     stop=None,
    # )
    # return res["choices"][0]["text"].strip()
    res: str = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': f"user", 'content': prompt}
        ],
        temperature=0,
    )
    print(res)
    return res["choices"][0]["message"]["content"].strip()


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
    # print("Data collected")
    return clean_string


def embed(query: str) -> str:
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


def upload(
        namespace: str, data: List[str], text_type: str = 'background', index_name: str = 'thesis-index'
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
        print(f"Index created: {index_name}")
    else:
        # print info about existing index
        index_info = pinecone.describe_index(INDEX_NAME)
        print(f"Index description: {index_info}")

    # print(namespace, text_type)
    # print(namespace_exist(namespace) and text_type == 'background')
    if namespace_exist(namespace) and text_type == 'background':
        return None

    # connect to pinecone and retrieve index
    index = pinecone.Index(INDEX_NAME)

    # wait for 30 seconds
    time.sleep(30)

    # create SentenceTransformer model and embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # upload data to pinecone index
    for j in tqdm(range(0, len(data), STRIDE)):
        j_end = min(j + WINDOW, len(data))  # get end of batch
        ids = [str(x) for x in range(j, j_end)]  # generate ID
        metadata = [{"text": text, "type": text_type} for text in data[j:j_end]]  # generate metadata
        embeddings = model.encode(data[j:j_end]).tolist()  # get embeddings
        curr_record = zip(ids, embeddings, metadata)  # compile into single vector
        index.upsert(vectors=curr_record, namespace=namespace)


def run_query_and_generate_answer(
        namespace: str,
        data: List[str],
        query: str,
        receiver: str,
        job: str,
        status: str,
        index_name: str,
        save: bool = True,
) -> str:
    """
    Runs a query on a Pinecone index and generates an answer based on the response context.
    :param namespace: The index namespace to operate in
    :param data: The data to be used for the query.
    :param query: The query to be run on the index.
    :param receiver: The character being prompted
    :param job: The profession of the receiver
    :param status: The social status of the character
    :param index_name: The name of the Pinecone index.
    :param save: A bool to save to a file is Ture and print out if False. Default: True
    :return: The generated answer based on the response context.
    """
    # connect to index
    index = pinecone.Index(INDEX_NAME)

    # upload data and generate query embedding.
    embedded_query = embed(
        query=query
    )

    upload(namespace, data, 'background', index_name)

    # query Pinecone index and get context for model prompting.
    responses = index.query(
        embedded_query, top_k=3, include_metadata=True, namespace=namespace, filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "answer"}}
            ]
        }
    )
    # print(responses)

    # Filter out responses containing the string "Player:"
    context = [x["metadata"]["text"] for x in responses["matches"] if query not in x["metadata"]["text"]]
    print(context)

    # generate clean prompt and answer.
    clean_prompt = prompt_engineer(query, receiver, job, status, context)
    save_prompt: str = clean_prompt.replace("\n", " ")
    # print(clean_prompt)
    # print(save_prompt)
    generated_answer = answer(clean_prompt, receiver)
    # print(generated_answer)
    update_history(namespace=namespace, info_file=DATA_FILE, prompt=query, response=generated_answer.split(": ")[-1])

    if save:
        # save results to file and return generated answer.
        with open("prompt_response.txt", "a") as save_file:
            save_file.write("\n" + "=" * 120 + "\n")
            save_file.write(f"Prompt: {save_prompt}\n")
            save_file.write(f"To: {receiver}, a {job}\n")
            clean_prompt = clean_prompt.replace('\n', ' ')
            save_file.write(f"{clean_prompt}\n{generated_answer}")
        print("Results saved to file.")
    else:
        print(generated_answer)

    return generated_answer


def update_history(namespace: str, info_file: str, prompt: str, response: str, index_name: str = 'thesis_index', character: str = "Player") -> None:
    # print(f"Question: {prompt}")
    # print(f"Answer: {response}")
    upload(namespace, [prompt], "question", index_name)
    upload(namespace, [response], "answer", index_name)

    with open(info_file, 'a') as history_file:
        history_file.write(f"\n{character}: {prompt}\nYou: {response}")


if __name__ == '__main__':
    GRAMMAR: dict = {
        'lower': 'poor',
        'middle': 'satisfactory',
        'high': 'formal'
    }
    WINDOW: int = 3  # how many sentences are combined
    STRIDE: int = 2  # used to create overlap, num sentences we 'stride' over
    QUERY: str = input("Insert question: ")
    # CHARACTER: str = "John Pebble"  # thief
    # CHARACTER: str = "Evelyn Stone-Brown"  # blacksmith
    # CHARACTER: str = "Caleb Brown"  # baker
    CHARACTER: str = 'Jack McCaster'  # fisherman
    # CHARACTER: str = "Peter Satoru"  # archer
    # CHARACTER: str = "Melinda Deek"  # knight
    # CHARACTER: str = "Sarah Ratengen"  # tavern owner

    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)
    PROFESSION, SOCIAL_CLASS = get_information(CHARACTER)
    print(CHARACTER, PROFESSION)
    DATA_FILE: str = f"Text Summaries/{names[CHARACTER]}.txt"

    INDEX_NAME: str = "thesis-index"
    NAMESPACE: str = extract_name(DATA_FILE).lower()

    file_data = load(DATA_FILE)

    openai.api_key = "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"  # windows 11
    pinecone.init(
        api_key="68b23f46-c793-40f6-a21e-dabbde2b3297",
        environment="northamerica-northeast1-gcp",
    )
    # pinecone.delete_index(INDEX_NAME)

    final_answer = run_query_and_generate_answer(
        namespace=NAMESPACE,
        data=file_data,
        receiver=CHARACTER,
        job=PROFESSION,
        status=SOCIAL_CLASS,
        query=QUERY,
        index_name=INDEX_NAME,
    )

    print(final_answer)
