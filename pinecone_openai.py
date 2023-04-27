import openai
from sentence_transformers import SentenceTransformer
import torch
import pinecone
from tqdm.auto import tqdm
import time

WINDOW: int = 3  # how many sentences are combined
STRIDE: int = 2  # used to create overlap, no. sentences we 'stride' over
DATA_FILE: str = "Text Summaries/john_pebble.txt"
INDEX_NAME: str = "thesis-index"

with open(DATA_FILE, "r") as text_file:  # get input and format data
    unclean = ""
    for line in text_file.readlines():
        unclean += line.strip("\n")
    data = [s.strip() + "." for s in unclean.split(".") if s.strip() != ""]

openai.api_key = "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"
pinecone.init(
    api_key="68b23f46-c793-40f6-a21e-dabbde2b3297",
    environment="northamerica-northeast1-gcp",
)

metadata_config = {"text": "query"}

if len(pinecone.list_indexes()) == 0:  # if no indices
    # create index
    pinecone.create_index(
        INDEX_NAME, dimension=384, metadata_config=metadata_config
    )  # make index
else:  # if indices already exist
    print(
        f"Index description: {pinecone.describe_index('thesis-index')}\n"
    )  # describe existing index

time.sleep(30)
index = pinecone.Index(INDEX_NAME)  # connect to index

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print(
        f"You are using {device}. This is much slower than using "
        "a CUDA-enabled GPU.\n"
    )

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

query = "What fish can be found near Ashbourne?"
embedded_query = model.encode(query).tolist()  # shape of (384, )

for i in tqdm(range(0, len(data), STRIDE)):
    i_end = min(i + WINDOW, len(data))  # get end of batch
    ids = [str(x) for x in range(i, i_end)]  # generate ID
    metadata = [{"text": text} for text in data[i:i_end]]  # generate metadata
    embeddings = model.encode(data[i:i_end]).tolist()  # get embeddings
    curr_record = zip(ids, embeddings, metadata)  # compile into single vector
    index.upsert(
        vectors=curr_record, namespace="john-pebble"
    )  # upsert to pinecone


# query pinecone
responses = index.query(
    embedded_query, top_k=3, include_metadata=True, namespace="john-pebble"
)
print(f"Prompt: {query}")
context = []
for result in responses["matches"]:
    context = result["metadata"]["text"]
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")

context = [x["metadata"]["text"] for x in responses["matches"]]
print(context)

pinecone.delete_index(INDEX_NAME)


def prompt_engineer(prompt, context) -> str:
    prompt_start = (
        "Answer the question based on the context below. "
        "Suggest a different person to visit if no answer is possible.\n\n"
        + "Context: "
    )
    prompt_end = f"\n\nQuestion: {prompt}\nAnswer: "
    prompt_middle = ""
    # append contexts until hitting limit
    for c in context:
        prompt_middle += f"\n{c}"
    return prompt_start + prompt_middle + prompt_end


def answer(prompt):
    res = openai.Completion.create(
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


clean_prompt = prompt_engineer(query, context)
print("----------")
print(clean_prompt)
generated_answer = answer(clean_prompt)
print(clean_prompt + generated_answer)
