import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from typing import List
import torch
from transformers import BertTokenizer, BertModel
import logging
import numpy as np
import pandas as pd
from ripser import ripser
from persim import plot_diagrams
import plotly.graph_objs as gobj


points = np.random.random((100, 2))
f = d.fill_rips(points, 2, 1.)
p = d.homology_persistence(f)
dgms = d.init_diagrams(p, f)


def plot_diagram(diagram, homology_dimensions=None, plotly_params=None):
    """Plot a single persistence diagram.

    Parameters
    ----------
    diagram : ndarray of shape (n_points, 3)
        The persistence diagram to plot, where the third dimension along axis 1
        contains homology dimensions, and the first two contain (birth, death)
        pairs to be used as coordinates in the two-dimensional plot.

    homology_dimensions : list of int or None, optional, default: ``None``
        Homology dimensions which will appear on the plot. If ``None``, all
        homology dimensions which appear in `diagram` will be plotted.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"traces"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing the persistence diagram.

    """
    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])

    diagram = diagram[diagram[:, 0] != diagram[:, 1]]
    diagram_no_dims = diagram[:, :2]
    posinfinite_mask = np.isposinf(diagram_no_dims)
    neginfinite_mask = np.isneginf(diagram_no_dims)
    max_val = np.max(np.where(posinfinite_mask, -np.inf, diagram_no_dims))
    min_val = np.min(np.where(neginfinite_mask, np.inf, diagram_no_dims))
    parameter_range = max_val - min_val
    extra_space_factor = 0.02
    has_posinfinite_death = np.any(posinfinite_mask[:, 1])
    if has_posinfinite_death:
        posinfinity_val = max_val + 0.1 * parameter_range
        extra_space_factor += 0.1
    extra_space = extra_space_factor * parameter_range
    min_val_display = min_val - extra_space
    max_val_display = max_val + extra_space

    fig = gobj.Figure()
    fig.add_trace(gobj.Scatter(
        x=[min_val_display, max_val_display],
        y=[min_val_display, max_val_display],
        mode="lines",
        line={"dash": "dash", "width": 1, "color": "black"},
        showlegend=False,
        hoverinfo="none"
        ))

    for dim in homology_dimensions:
        name = f"H{int(dim)}" if dim != np.inf else "Any homology dimension"
        subdiagram = diagram[diagram[:, 2] == dim]
        unique, inverse, counts = np.unique(
            subdiagram, axis=0, return_inverse=True, return_counts=True
            )
        hovertext = [
            f"{tuple(unique[unique_row_index][:2])}" +
            (
                f", multiplicity: {counts[unique_row_index]}"
                if counts[unique_row_index] > 1 else ""
            )
            for unique_row_index in inverse
            ]
        y = subdiagram[:, 1]
        if has_posinfinite_death:
            y[np.isposinf(y)] = posinfinity_val
        fig.add_trace(gobj.Scatter(
            x=subdiagram[:, 0], y=y, mode="markers",
            hoverinfo="text", hovertext=hovertext, name=name
        ))

    fig.update_layout(
        width=500,
        height=500,
        xaxis1={
            "title": "Birth",
            "side": "bottom",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            },
        yaxis1={
            "title": "Death",
            "side": "left",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False, "scaleanchor": "x", "scaleratio": 1,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            },
        plot_bgcolor="white"
        )

    # Add a horizontal dashed line for points with infinite death
    if has_posinfinite_death:
        fig.add_trace(gobj.Scatter(
            x=[min_val_display, max_val_display],
            y=[posinfinity_val, posinfinity_val],
            mode="lines",
            line={"dash": "dash", "width": 0.5, "color": "black"},
            showlegend=True,
            name=u"\u221E",
            hoverinfo="none"
        ))

    # Update traces and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig


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

# e = [get_bert_embeddings(s) for s in sentences]
#
# # Convert the list of BERT embeddings to a numpy array
# embeddings = np.array(e).T
# print(embeddings.shape)
#
# # Compute persistent homology using ripser
# # Compute persistent homology using ripser
# result = ripser(embeddings)
# diagrams = result['dgms']
#
# # Create a figure with subplots
# fig, axes = plt.subplots(len(diagrams), 1, figsize=(8, 6), sharex=True)
#
# # Iterate over all dimensions
# for dim, features in enumerate(diagrams):
#     # Extract birth and death values
#     birth_values = [point[0] for point in features]
#     death_values = [point[1] for point in features]
#     sorted_death = sorted(death_values, reverse=True)
#     print(sorted_death)
#
#     # Compute maximum epsilon value
#     for i in sorted_death:
#         if i != np.inf:
#             max_epsilon = i
#             break
#
#     # Plot the bar chart in the corresponding subplot
#     ax = axes[dim]
#     ax.bar(range(len(birth_values)), height=[point[1] - point[0] for point in features],
#            align='center', alpha=0.5, color='blue', label=f'H{dim} Features')
#     ax.set_ylabel('Persistence')
#     ax.set_title(f'Persistent Homology H{dim} Bar Chart')
#     ax.set_ylim(0, max_epsilon)  # Set y-axis limits
#
# # Set the x-axis label for the last subplot
# axes[-1].set_xlabel('Feature Index')
#
# # Adjust spacing between subplots
# plt.tight_layout()
#
# # Show the combined figure
# plt.show()

# # Extract birth and death values from the persistence diagrams
# birth_values = [point[0] for point in diagrams[0]]
# death_values = [point[1] for point in diagrams[0]]
#
# # Create a scatter plot of the persistence diagram using Plotly
# fig = go.Figure(data=go.Scatter(x=birth_values, y=death_values, mode='markers'))
#
# # Customize the plot appearance
# fig.update_layout(
#     title='Persistence Diagram',
#     xaxis_title='Birth',
#     yaxis_title='Death',
# )
#
# # Display the plot
# fig.show()

# figure = plot_diagram(list(zip(birth_values, death_values)))


"""
# Generate a random point cloud
np.random.seed(0)
point_cloud = np.random.random((100, 2))
print(point_cloud)

# Compute persistent homology
diagrams = ripser(point_cloud)['dgms']
"""
# Plot the persistence diagrams
# plot_diagrams(diagrams, show=True)
# """

# import networkx as nx
#
# # Choose a filtration parameter (e.g., maximum distance)
# filtration_parameter = 0.5
#
# # Construct the simplicial complex
# G = nx.Graph()
# for dim, diagram in enumerate(diagrams):
#     for point in diagram:
#         birth, death = point
#         if death - birth > filtration_parameter:
#             continue
#         G.add_node(point)
#         G.nodes[point]['birth'] = birth
#         G.nodes[point]['death'] = death
#
# # Add edges based on connectivity
# for edge in simplicial_complex:
#     G.add_edge(edge[0], edge[1])
#
# # Visualize the graph
# nx.draw(G, with_labels=True, node_size=200, font_size=8, edge_color='gray', node_color='lightblue')
# plt.show()
