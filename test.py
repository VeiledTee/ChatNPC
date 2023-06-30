import numpy as np
from ripser import ripser
from persim import plot_diagrams
from typing import List
import torch
from transformers import BertTokenizer, BertModel
import logging
import numpy as np
import pandas as pd
from ripser import ripser
from persim import plot_diagrams
import plotly.graph_objs as gobj


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

data: pd.DataFrame = pd.read_csv("Data/contrast-dataset.csv")

e = [get_bert_embeddings(s) for s in data["Phrase"].values]

# Convert the list of BERT embeddings to a numpy array
embeddings = np.array(e).T
print(embeddings.shape)

# Compute persistent homology using ripser
diagrams = ripser(embeddings)['dgms']

print(len(diagrams))
print(diagrams[0].shape)

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
plot_diagrams(diagrams, show=True)
# """
