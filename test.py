import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from typing import List, Tuple, Any
import torch
from transformers import BertTokenizer, BertModel
import logging
import numpy as np
import pandas as pd
from ripser import ripser
from persim import plot_diagrams
import plotly.graph_objs as gobj


def top_k_holes(diagrams, k: int = 3):
    print(len(diagrams[0]))
    # Iterate over each dimension
    for dim, diagram in enumerate(diagrams):
        # Initialize an empty list to store the hole indices and their persistence values
        holes = []
        print(dim)
        # Iterate over each feature in the diagram
        for i, (birth, death) in enumerate(diagram):
            persistence = death - birth
            holes.append((dim, i, persistence))

        # Sort the holes based on their persistence values in descending order
        holes.sort(key=lambda x: x[2], reverse=True)

        # Select the top k holes
        top_k = holes[:k]

        # Print the selected holes
        for hole in top_k:
            dim, index, persistence = hole
            print(f"Dimension: {dim}, Hole Index: {index}, Persistence: {persistence}")


def get_bert_embeddings(sentence: str) -> List[float]:
    # Disable the logging level for the transformers library
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Restoring the logging level
    logging.getLogger("transformers").setLevel(logging.INFO)

    # Tokenize and encode the sentence
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor([tokens])

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

    # Convert embeddings to a Python list
    embeddings_list = embeddings.tolist()

    return embeddings_list


def get_bert_tokens_embeddings(sentence: str) -> tuple[list[str], torch.Tensor]:
    # Disable the logging level for the transformers library
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Restoring the logging level
    logging.getLogger("transformers").setLevel(logging.INFO)

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Define the desired length of the embeddings
    max_length = 128

    # Pad or truncate token IDs to the desired length
    if len(token_ids) < max_length:
        token_ids = token_ids + [0] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    # Convert token IDs to tensor
    token_ids_tensor = torch.tensor([token_ids])

    # Get the token embeddings
    with torch.no_grad():
        outputs = model(token_ids_tensor)
        token_embeddings = outputs.last_hidden_state

    # Convert token embeddings to a list
    token_embeddings = token_embeddings.squeeze().tolist()

    return tokens, token_embeddings



sentences = [
    "The sky is blue.",
    "I love eating pizza.",
    "She plays the piano beautifully.",
    "The cat is sleeping.",
    "I enjoy reading books.",
    "He runs every morning.",
    "The flowers are blooming in the garden.",
    "They went for a walk in the park.",
    "The movie was fantastic.",
    "We had a great time at the beach.",
    "She smiled and waved at me.",
    "The rain is pouring outside.",
    "He is studying for his exams.",
    "The coffee tastes delicious.",
    "I'm going to the gym later.",
    "They are planning a trip to Europe.",
    "She wrote a poem for her friend.",
    "He likes to watch football on weekends.",
    "The concert was amazing.",
    "We had a delicious dinner at the restaurant.",
    "Fuck",
    "Shit",
    "Cunt",
]

# sentences = [
#     "Billy loves cake",
#     "Olivia loves cake",
#     "Josh hates cake"
# ]

# data: pd.DataFrame = pd.read_csv("Data/contrast-dataset.csv")
# sentences = data["Phrase"].values

e = [get_bert_embeddings(s) for s in sentences]
# t, e = zip(*[get_bert_embeddings(s) for s in sentences])

# Convert the list of BERT embeddings to a numpy array
embeddings = np.array(e).T

# Compute persistent homology using ripser
result = ripser(embeddings)
diagrams = result['dgms']

top_k_holes(diagrams, 8)
# print(diagrams)

# hole_durations = []  # List to store persistence durations
#
# for feature in diagrams:
#     for element in feature:
#         if any(e == float('inf') for e in element):
#             continue  # Skip if any element is infinity
#         else:
#             birth = element[0]
#             death = element[1]
#             duration = death - birth
#             hole_durations.append(duration)
#
# # Print the persistence durations of the holes
# for i, duration in enumerate(hole_durations):
#     print(f"Hole {i+1}: Persistence Duration = {duration}")


# Create a figure with subplots
fig, axes = plt.subplots(len(diagrams), 1, figsize=(8, 6), sharex=True)

# Iterate over all dimensions
for dim, features in enumerate(diagrams):
    # Extract birth and death values
    birth_values = [point[0] for point in features]
    death_values = [point[1] for point in features]
    sorted_death = sorted(death_values, reverse=True)
    print(sorted_death)

    # Compute maximum epsilon value
    for i in sorted_death:
        if i != np.inf:
            max_epsilon = i
            break

    # Plot the bar chart in the corresponding subplot
    ax = axes[dim]
    ax.bar(range(len(birth_values)), height=[point[1] - point[0] for point in features],
           align='center', alpha=0.5, color='blue', label=f'H{dim} Features')
    ax.set_ylabel('Persistence')
    ax.set_title(f'Persistent Homology H{dim} Bar Chart')
    ax.set_ylim(0, max_epsilon)  # Set y-axis limits

# Set the x-axis label for the last subplot
axes[-1].set_xlabel('Feature Index')

# Adjust spacing between subplots
plt.tight_layout()

# Show the combined figure
plt.show()

dimensions: list[list[float]] = []
for dim, diagram in enumerate(diagrams):
    print(f"Dimension {dim}:")
    x_values: list[float] = []
    widths: list[float] = []
    # Iterate over each feature in the diagram
    for i, (birth, death) in enumerate(diagram):
        x_values.append(birth)
        widths.append(death - birth)
    print(widths.index(max(widths)), max(widths))
    print(widths[widths.index(max(widths))])
    print(diagram[widths.index(max(widths))])
    # print(len(widths[widths.index(max(widths))]))
    plt.barh(x_values, widths, left=x_values, height=0.001)
    # Set the axis labels
    plt.xlabel('Duration')
    plt.ylabel('Birth')

    # Show the plot
    plt.show()
    dimensions.append([x_values, widths])


"""
Get longest lasting feature
print and figure out shape
"""
