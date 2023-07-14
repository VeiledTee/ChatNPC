import logging
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from ripser import ripser
from transformers import BertTokenizer, BertModel

# Check if GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device('cpu')
    print("GPU is not available. PyTorch is using CPU.")

# Disable the logging level for the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)
matplotlib.use("TkAgg")


def get_bert_embeddings(sentence: str) -> np.ndarray:
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

    # Tokenize and encode the sentence
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor([tokens]).to(DEVICE)

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        word_embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

    # Convert embeddings to a NumPy aray
    return word_embeddings.cpu().numpy()


def top_k_holes(ph_diagrams: List[np.ndarray], k: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    Find the k holes in each dimension that persist the longest and return their information
    :param ph_diagrams: Diagrams generated using the "ripser" library
    :param k: A list of k values to use in each dimension.
    :return: List of top k holes in each dimension
    """
    if k is None:
        k = [260, 50]
    if len(k) > len(ph_diagrams):
        print(Warning(
            f"You provided more k values than dimensions. There are {len(ph_diagrams)} dimensions but {len(k)} k values. Only the first {len(ph_diagrams)} k values will be used"))
    elif len(k) < len(ph_diagrams):
        raise ValueError(
            f"Less k values than there are dimensions. You provided {len(k)} k values for {len(ph_diagrams)} dimensions. Ensure there is a k value for every dimension.")

    top_holes: List[np.ndarray] = []
    # Iterate over each dimension
    for dimension, diag_array in enumerate(ph_diagrams):
        # Initialize an empty list to store the hole indices and their persistence values
        holes = []
        # Iterate over each feature in the diagram
        for j in range(diag_array.shape[0]):
            feature_birth = diag_array[j, 0]
            feature_death = diag_array[j, 1]
            persistence = feature_death - feature_birth
            holes.append(np.array([dimension, j, feature_birth, feature_death, persistence]))

        # Sort the holes based on their persistence values in descending order
        holes.sort(key=lambda x: x[4], reverse=True)

        # Select the top k holes and add to list
        top_holes.append(np.array(holes[:k[dimension]]))

    # return list of top k holes in each dimension
    return top_holes


def separate_ph(ph_diagrams):
    dimensions = []
    for dimension, diag_tuple in enumerate(ph_diagrams):
        x_values: list[float] = []
        widths: list[float] = []
        # Iterate over each feature in the diagram
        for j, (feature_birth, feature_death) in enumerate(diag_tuple):
            x_values.append(feature_birth)
            widths.append(feature_death - feature_birth)

        plt.barh(x_values, widths, left=x_values, height=0.001)
        # Set the axis labels
        plt.xlabel("Duration")
        plt.ylabel("Birth")

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
        ax.barh(
            range(len(birth_values)),
            width=[point[1] - point[0] for point in f],
            left=birth_values,
            align="center",
            alpha=0.5,
            color="blue",
            label=f"H{d} Features",
        )
        ax.set_xlabel("\u03B5 value")
        ax.set_title(f"Persistent Homology H{d} Bar Chart")
        ax.set_xlim(0, max_epsilon)  # Set x-axis limits

    # Set the y-axis label for the last subplot
    axes[-1].set_ylabel("Feature Index")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the combined figure
    plt.show()


def persistent_homology_features(phrases: List[str]) -> List[List[np.ndarray]]:
    features: list = []
    for sentence in phrases:
        embedding = [get_bert_embeddings(s) for s in sentence]

        # Convert the list of BERT embeddings to a numpy array
        all_embeddings = np.array(embedding).T

        # Compute persistent homology using ripser
        ph = ripser(all_embeddings, maxdim=1)
        ph_diagrams = ph["dgms"]

        phrase_holes: list = top_k_holes(ph_diagrams)
        for number, dimension in enumerate(phrase_holes):
            print(f"Dimension {number} Len: {len(dimension)}")
            print(f"Dimension {number} shape: {dimension.shape}")
        print()
        features.append(phrase_holes)
    return features


# sentences = [
#     "The sky is blue.",
#     "I love eating pizza.",
#     "She plays the piano beautifully.",
#     "The cat is sleeping.",
#     "I enjoy reading books.",
#     "He runs every morning.",
#     "The flowers are blooming in the garden.",
#     "They went for a walk in the park.",
#     "The movie was fantastic.",
#     "We had a great time at the beach.",
#     "She smiled and waved at me.",
#     "The rain is pouring outside.",
#     "He is studying for his exams.",
#     "The coffee tastes delicious.",
#     "I'm going to the gym later.",
#     "They are planning a trip to Europe.",
#     "She wrote a poem for her friend.",
#     "He likes to watch football on weekends.",
#     "The concert was amazing.",
#     "We had a delicious dinner at the restaurant.",
# ]

sentences = ["Billy loves cake", "Josh hates cake"]

ph_features: list = persistent_homology_features(phrases=sentences)

# data: pd.DataFrame = pd.read_csv("Data/contrast-dataset.csv")
# sentences = data["Phrase"].values
#
# for phrase in sentences:
#     e = [get_bert_embeddings(s) for s in phrase]
#
#     # Convert the list of BERT embeddings to a numpy array
#     embeddings = np.array(e).T
#
#     # Compute persistent homology using ripser
#     result = ripser(embeddings, maxdim=1)
#     diagrams = result["dgms"]
#
#     k_holes: list = top_k_holes(diagrams)
#     for dim in k_holes:
#         print(f"Dim Len: {len(dim)}")
#         print(f"Dim shape: {dim.shape}")
#         for hole in dim:
#             hole_dim, index, birth, death, persist = hole
#             # print(f"Dimension: {hole_dim}, Hole Index: {index}, Birth: {birth}, Persistence: {persist}")
#             # print([hole[i] for i in [0, 2, 3]])
#     print()
#     # plot_ph_across_dimensions(diagrams)
#
#     # print(diagrams)
#
#     # hole_durations = []  # List to store persistence durations
#     #
#     # for feature in diagrams:
#     #     for element in feature:
#     #         if any(e == float('inf') for e in element):
#     #             continue  # Skip if any element is infinity
#     #         else:
#     #             birth = element[0]
#     #             death = element[1]
#     #             duration = death - birth
#     #             hole_durations.append(duration)
#     #
#     # # Print the persistence durations of the holes
#     # for i, duration in enumerate(hole_durations):
#     #     print(f"Hole {i+1}: Persistence Duration = {duration}")
#
#     # Create a figure with subplots
#
#     """
#     Get longest lasting feature
#     print and figure out shape
#     """
