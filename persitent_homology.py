import logging
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from ripser import ripser
from transformers import BertTokenizer, BertModel

matplotlib.use('TkAgg')


def top_k_holes(ph_diagrams, k: int = 3):
    top_holes: list = []
    # Iterate over each dimension
    for dimension, diag_tuple in enumerate(ph_diagrams):
        # Initialize an empty list to store the hole indices and their persistence values
        holes = []
        print(dimension)
        # Iterate over each feature in the diagram
        for j, (feature_birth, feature_death) in enumerate(diag_tuple):
            persistence = feature_death - feature_birth
            holes.append((dimension, j, persistence))

        # Sort the holes based on their persistence values in descending order
        holes.sort(key=lambda x: x[2], reverse=True)

        # Select the top k holes and add to list
        top_holes.append(holes[:k])

    # return list of top k holes in each dimension
    return top_holes


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
        word_embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

    # Convert embeddings to a Python list
    embeddings_list = word_embeddings.tolist()

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


def separate_ph(ph_diagrams):
    dimensions = []
    for dimension, diag_tuple in enumerate(ph_diagrams):
        print(f"Dimension {dimension}:")
        x_values: list[float] = []
        widths: list[float] = []
        # Iterate over each feature in the diagram
        for j, (feature_birth, feature_death) in enumerate(diag_tuple):
            x_values.append(feature_birth)
            widths.append(feature_death - feature_birth)

        plt.barh(x_values, widths, left=x_values, height=0.001)
        # Set the axis labels
        plt.xlabel('Duration')
        plt.ylabel('Birth')

        # Show the plot
        plt.show()
        dimensions.append([x_values, widths])


def plot_ph_across_dimensions(ph_diagrams):
    fig, axes = plt.subplots(len(ph_diagrams), 1, figsize=(10, 8))

    # Iterate over all dimensions
    for d, f in enumerate(ph_diagrams):
        # Extract birth and death values
        birth_values = [point[0] for point in f]
        death_values = [point[1] for point in f]
        sorted_death = sorted(death_values, reverse=True)

        # Compute maximum epsilon value
        for i in sorted_death:
            if i != np.inf:
                max_epsilon = i
                break

        # Plot the horizontal bar chart in the corresponding subplot
        ax = axes[d]
        ax.barh(range(len(birth_values)), width=[point[1] - point[0] for point in f], left=birth_values,
                align='center', alpha=0.5, color='blue', label=f'H{d} Features')
        ax.set_xlabel('\u03B5 value')
        ax.set_title(f'Persistent Homology H{d} Bar Chart')
        ax.set_xlim(0, max_epsilon)  # Set x-axis limits

    # Set the y-axis label for the last subplot
    axes[-1].set_ylabel('Feature Index')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the combined figure
    plt.show()


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
result = ripser(embeddings, maxdim=2)
diagrams = result['dgms']

k_holes: list = top_k_holes(diagrams, 8)
for dim in k_holes:
    for hole in dim:
        hole_dim, index, persist = hole
        print(f"Dimension: {hole_dim}, Hole Index: {index}, Persistence: {persist}")

plot_ph_across_dimensions(diagrams)

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

"""
Get longest lasting feature
print and figure out shape
"""
