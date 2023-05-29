import json
from typing import List

import openai
import pinecone
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def extract_name(file_name: str) -> str:
    """
    Extracts the name of a character from their descriptions file name
    :param file_name: the file containing the character's description
    :return: the character's name, separated by a hyphen
    """
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
    """
    Check if a namespace exists in Pinecone index
    :param namespace: the namespace in question
    :return: boolean showing if the namespace exists or not
    """
    index = pinecone.Index("thesis-index")  # get index
    responses = index.query(
        embed(" "),
        top_k=1,
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "response"}},
            ]
        },
    )  # query index
    return (
        responses["matches"] != []
    )  # if matches comes back empty namespace doesn't exist


def get_information(character_name) -> None | tuple:
    """
    Return information from the character's data file
    :param character_name: name of character
    :return: None if character doesn't exist | profession and social status if they do exist
    """
    with open("Text Summaries/character_objects.json", "r") as f:
        data = json.load(f)

    for character in data["characters"]:
        if character["name"].strip() == character_name.strip():
            return character["profession"], character["social status"]

    return None  # character not found


def prompt_engineer(prompt: str, status: str, context: List[str]) -> str:
    """
    Given a base query and context, format it to be used as prompt
    :param job:
    :param receiver: The character being prompted
    :param prompt: The prompt query
    :param context: The context to be used in the prompt
    :return: The formatted prompt
    """
    prompt_start = (
        f"Use {GRAMMAR[status.split()[0]]} grammar. Use first person. "
        f"Reply in a single, clear sentence based on the context. When told "
        f"new information, rephrase the information as a fact. "
        f"Do not mention your background or the context unless asked, or that you are fictional. "
        f"Do not provide facts you would deny. Context:"
    )
    with open("tried_prompts.txt", "a") as prompt_file:
        if prompt_start not in prompt_file.readlines():
            prompt_file.write(prompt_start + "\n")
    prompt_end = f"\n\nQuestion: {prompt}\nAnswer: "
    prompt_middle = ""
    # append contexts until hitting limit
    for c in context:
        prompt_middle += f"\n{c}"
    return prompt_start + prompt_middle + prompt_end


def generate_conversation(
    character_file: str, player: bool, next_phrase: str
) -> None:
    """
    Generate a record of the conversation a user has had with the system for feeding into gpt 3.5 turbo
    :param character_file: The file associated with a character's background, not required unless first execution of function to 'set the stage'
    :return: A list of dictionaries
    {"role": "user", "content": "Where was it played?"}
    """
    if not HISTORY:
        with open(character_file) as char_file:
            background: str = f"You are {CHARACTER}. Your background:"
            for line in char_file.readlines():
                background += " " + line.strip()
        HISTORY.append({"role": "system", "content": background})
    if player:
        HISTORY.append({"role": "user", "content": next_phrase})
    else:
        HISTORY.append({"role": "assistant", "content": next_phrase})


def answer(prompt: str, chat_history: List[dict], chat: bool = True) -> str:
    """
    Using openAI API, respond ot the provide prompt
    :param prompt: An engineered prompt to get the language model to respond to
    :param chat_history:
    :return: The completed prompt
    """
    if chat:
        msgs: List[dict] = chat_history
        msgs.append(
            {"role": "user", "content": prompt}
        )  # build current history of conversation for model
        res: str = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=msgs,
            temperature=0,
        )

        return res["choices"][0]["message"][
            "content"
        ].strip()  # get model response
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
        )
        return res["choices"][0]["text"].strip()


def load_file_information(load_file: str) -> List[str]:
    """
    Loads data from a file into a list of strings for embedding.
    :param load_file: Path to a file containing the data we want to encrypt.
    :return: A list of all the sentences in the load_file.
    """
    with open(load_file, "r") as text_file:
        # list comprehension to clean and split the text file
        clean_string = [
            s.strip() + "."
            for line in text_file
            for s in line.strip().split(".")
            if s.strip()
        ]
    return clean_string


def embed(query: str) -> str:
    """
    Take a sentence of text and return the 384-dimension embedding
    :param query: The sentence to be embedded
    :return: Embedding representation of the sentence
    """
    # create SentenceTransformer model and embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast and good
    # model = SentenceTransformer("")  # slow and great
    device = "cuda" if torch.cuda.is_available() else "cpu"  # gpu check
    model = model.to(device)
    return model.encode(query).tolist()


def upload(
    namespace: str,
    data: List[str],
    text_type: str = "background",
    index_name: str = "thesis-index",
) -> None:
    """
    Upserts text embedding vectors into pinecone DB at the specific index
    :param data: Data to be embedded and stored
    :param text_type: The type of text we are embedding. Choose "background", "response", or "question". Default value: "background"
    :param index_name: the name of the pinecone index to save the data to
    """
    if not pinecone.list_indexes():  # check if there are any indexes
        # create index if it doesn't exist
        pinecone.create_index(index_name, dimension=384)

    if namespace_exist(namespace) and text_type == "background":
        return None

    # connect to pinecone and retrieve index
    index = pinecone.Index(INDEX_NAME)

    # create SentenceTransformer model and embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = index.query(embed(" "), top_k=10000, namespace=namespace)

    # upload data to pinecone index
    for j in tqdm(range(0, len(data), STRIDE)):
        j_end = min(j + WINDOW, len(data))  # get end of batch
        ids = [
            str(len(results["matches"])) for _ in range(j, j_end)
        ]  # generate ID
        metadata = [
            {"text": text, "type": text_type} for text in data[j:j_end]
        ]  # generate metadata
        embeddings = model.encode(data[j:j_end]).tolist()  # get embeddings
        curr_record = zip(
            ids, embeddings, metadata
        )  # compile into single vector
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
    generate_conversation(
        f"Text Summaries/{namespace.replace('-', '_')}.txt", True, query
    )

    # connect to index
    index = pinecone.Index(INDEX_NAME)

    # upload data and generate query embedding.
    embedded_query = embed(query=query)

    upload(namespace, data, "background", index_name)

    # query Pinecone index and get context for model prompting.
    responses = index.query(
        embedded_query,
        top_k=3,
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "response"}},
                # {"type": {"$eq": "query"}},
            ]
        },
    )

    # Filter out responses containing the string "Player:"
    context = [
        x["metadata"]["text"]
        for x in responses["matches"]
        if query not in x["metadata"]["text"]
    ]

    # generate clean prompt and answer.
    clean_prompt = prompt_engineer(query, status, context)
    save_prompt: str = clean_prompt.replace("\n", " ")

    generated_answer = answer(clean_prompt, HISTORY)

    if save:
        # save results to file and return generated answer.
        with open("prompt_response.txt", "a") as save_file:
            save_file.write("\n" + "=" * 120 + "\n")
            save_file.write(f"Prompt: {save_prompt}\n")
            save_file.write(f"To: {receiver}, a {job}\n")
            clean_prompt = clean_prompt.replace("\n", " ")
            save_file.write(f"{clean_prompt}\n{generated_answer}")
        print("Results saved to file.")
    else:
        print(generated_answer)

    generate_conversation(
        f"Text Summaries/{namespace.replace('-', '_')}.txt",
        False,
        generated_answer,
    )

    update_history(
        namespace=namespace,
        info_file=DATA_FILE,
        prompt=query,
        response=generated_answer.split(": ")[-1],
    )

    return generated_answer


def update_history(
    namespace: str,
    info_file: str,
    prompt: str,
    response: str,
    index_name: str = "thesis_index",
    character: str = "Player",
) -> None:
    """
    Update the history of the current chat with new responses
    :param namespace: namespace of character we are talking to
    :param info_file: file where chat history is logged
    :param prompt: prompt user input
    :param response: response given by LLM
    :param index_name: name of the index associated with the namespace
    :param character: the character we are conversing with
    """
    upload(namespace, [prompt], "query", index_name)
    upload(namespace, [response], "response", index_name)

    extension_index = info_file.index(".")
    new_filename = (
        info_file[:extension_index] + "_chat" + info_file[extension_index:]
    )

    with open(new_filename, "a") as history_file:
        history_file.write(f"{character}: {prompt}\nYou: {response}\n")


def chat(
    namespace: str,
    data: List[str],
    receiver: str,
    job: str,
    status: str,
    index_name: str,
) -> None:
    """
    Initiate a conversation with a character. Stops conversation when player says "bye".
    :param namespace: Namespace of the character, reference to knowledge base
    :param data: Text data
    :param receiver: The character receiving query
    :param job: The character's profession
    :param status: Social status of character
    :param index_name: The index name of the pinecone index
    :return: None
    """
    while True:
        QUERY: str = input("Player: ")

        if QUERY.lower() == "bye":
            break

        final_answer = run_query_and_generate_answer(
            namespace=namespace,
            data=data,
            receiver=receiver,
            job=job,
            status=status,
            query=QUERY,
            index_name=index_name,
        )

        print(f"{receiver}: {final_answer}")


if __name__ == "__main__":
    GRAMMAR: dict = {
        "lower": "poor",
        "middle": "satisfactory",
        "high": "formal",
    }
    WINDOW: int = 3  # how many sentences are combined
    STRIDE: int = 2  # used to create overlap, num sentences we 'stride' over

    HISTORY: List[dict] = []

    # CHARACTER: str = "John Pebble"  # thief
    # CHARACTER: str = "Evelyn Stone-Brown"  # blacksmith
    # CHARACTER: str = "Caleb Brown"  # baker
    # CHARACTER: str = 'Jack McCaster'  # fisherman
    # CHARACTER: str = "Peter Satoru"  # archer
    CHARACTER: str = "Melinda Deek"  # knight
    # CHARACTER: str = "Sarah Ratengen"  # tavern owner

    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    PROFESSION, SOCIAL_CLASS = get_information(CHARACTER)
    print(CHARACTER, PROFESSION)
    DATA_FILE: str = f"Text Summaries/{names[CHARACTER]}.txt"

    INDEX_NAME: str = "thesis-index"
    NAMESPACE: str = extract_name(DATA_FILE).lower()

    file_data = load_file_information(DATA_FILE)

    with open('keys.txt', 'r') as key_file:
        openai.api_key = (key_file.readlines()[0])
        pinecone.init(api_key=key_file.readlines()[1], environment=key_file.readlines()[2])

    chat(
        namespace=NAMESPACE,
        data=file_data,
        receiver=CHARACTER,
        job=PROFESSION,
        status=SOCIAL_CLASS,
        index_name=INDEX_NAME,
    )
