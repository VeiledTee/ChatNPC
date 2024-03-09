import pandas as pd
import numpy as np
from webchat import embed
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine
import itertools
import plotly.express as px
import plotly.graph_objects as go


def cosine_similarity(vector_a: list, vector_b: list) -> float:
    """
    Finds the cosine similarity of two vectors
    :param vector_a: The first vector to compare
    :param vector_b: The second vector to compare
    :return: The cosine similarity score of the two vectors
    """
    return 1 - cosine(vector_a, vector_b)


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode unique topic strings as integers
    :param df: Dataframe containing raw topics
    :return: Altered dataframe
    """
    topics: list = list(df["Topic"].unique())
    mapping: dict = dict(zip(topics, range(len(topics))))
    df["Topic Num"] = [mapping[topic] for topic in df["Topic"].values]
    print(mapping)
    return df


def generate_tsne_embeddings(fit_on, dimensions: int = 2) -> np.ndarray:
    """
    Create embeddings of the specific dimension from data passed to the function
    :param fit_on: Data to fit T-SNE on
    :param dimensions: Num dimensions to reduce to
    :return: Generated embeddings
    """
    return TSNE(
        n_components=dimensions,
        learning_rate="auto",
        init="random",
        perplexity=30,
        n_iter=1000,
        n_jobs=-1,
        random_state=42,
    ).fit_transform(fit_on)


def display_tsne(x_col: str, y_col: str, df: pd.DataFrame) -> None:
    """
    Display T-SNE embedding data in scatterplot
    :param x_col: Name of column in df containing x-values of T-SNE embedding
    :param y_col: Name of column in df containing y-values of T-SNE embedding
    :param df: Dataframe object holding the T-SNE embeddings of original text
    :return: None
    """
    fig = px.scatter(
        data_frame=df, x=x_col, y=y_col, hover_data=["ID"], color="Topic", symbol="Topic", title="T-SNE Representation"
    )
    fig.update_traces(hovertemplate="<br>ID=%{customdata[0]}")
    fig.update_layout(hovermode="closest")
    fig.show()


def visualize_heatmap(data_to_visualize, axis_labels, title: str) -> None:
    """
    Visualize a matrix ias a heatmap
    :param data_to_visualize: A 2d matrix representing the value of each cell in the final heatmap
    :param axis_labels: Labels for the y axis
    :param title: Title of the plot
    :return: None
    """
    scale = int(20 * len(axis_labels))  # find scale
    fig = px.imshow(data_to_visualize, title=title)  # generate heatmap
    tick_values = []
    for i in range(0, scale + len(axis_labels), int(scale / len(axis_labels))):  # generate grid over heatmap
        tick_values.append(i)
        if i >= 0:
            fig.add_trace(
                go.Scatter(
                    x=[i - 0.5, i - 0.5],
                    y=[-0.5, scale - 0.5],
                    mode="lines",
                    line_color="black",
                    line_width=1,
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[-0.5, scale - 0.5],
                    y=[i - 0.5, i - 0.5],
                    mode="lines",
                    line_color="black",
                    line_width=1,
                    showlegend=False,
                )
            )
    fig.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=[i + 9.5 for i in tick_values], ticktext=axis_labels, side="bottom", tickangle=45
        ),
        yaxis=dict(tickmode="array", tickvals=[i + 9.5 for i in tick_values], ticktext=axis_labels),
    )  # format lables
    fig.show()


def average_similarity(df: pd.DataFrame, label_column: str = "Topic Num") -> dict:
    """
    Finds the average similarity of each topics phrases to each other
    :param df: Dataframe containing the embeddings and their labels
    :param label_column: Name of the column the labels are contained within
    :return: A dictionary stating the average cosine similarity score for each topic present in the dataset
    """
    averages: dict = {}
    for i in range(len(data[label_column].unique())):
        topic_embeddings = np.array(list(df.loc[data[label_column] == i, "Embedding"].values))
        avg: float = 0
        comparisons: int = 0
        for j, k in itertools.combinations(topic_embeddings, 2):
            avg += cosine_similarity(j, k)
            comparisons += 1
        averages[f"{i}"] = avg / comparisons
    return averages


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Given a list of vectors, find the cosine similarity between all of them and create a 2d matrix housing the values
    :param vectors: The list of vectors to analyze
    :return: Matrix of cosine similarities
    """
    # print(len(vectors[0]))  # 768
    similarity_matrix: list = []
    for i in range(len(vectors)):
        curr: list = []
        for j in range(len(vectors)):
            curr.append(cosine_similarity(vectors[i], vectors[j]))
        similarity_matrix.append(curr)
    return np.array(similarity_matrix)


def kmeans_cluster(x_col: str, y_col: str, embedding_col: str, df: pd.DataFrame, num_clusters: int) -> None:
    """
    Execute K-Means clustering on data based on the number of clusters presented
    :param x_col: Name of the column housing the x-data
    :param y_col: Name of the column housing the y-data
    :param embedding_col: Name of the column housing the embeddings of the data
    :param df: Dataframe containing all data
    :param num_clusters: Number of clusters to assign
    :return: None
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_data = list(df[embedding_col].values)
    kmeans.fit(kmeans_data)
    df["Cluster"] = kmeans.labels_
    fig = px.scatter(
        data_frame=df,
        x=x_col,
        y=y_col,
        hover_data=["Topic", "ID", "Cluster"],
        color="Topic",
        symbol="Cluster",
        hover_name="ID",
        title="K-Means Clustering",
    )
    fig.update_traces(hovertemplate="<br>Topic=%{customdata[0]}<br>ID=%{customdata[1]}<br>Cluster=%{customdata[2]}")
    fig.add_trace(
        go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode="markers",
            hovertemplate="Centroid",
            name="Centroid",
            showlegend=True,
        )
    )
    fig.update_layout(hovermode="closest")
    fig.show()


if __name__ == "__main__":
    # preprocessing
    data: pd.DataFrame = pd.read_csv("Data/contrast-dataset.csv")
    topic_labels: list = list(data["Topic"].values)
    data = one_hot_encoding(data)
    data["Embedding"] = data["Phrase"].apply(embed)
    phrase_embeddings: np.ndarray = np.array(list(data["Embedding"].values))
    # Distance matrix analysis
    euclid_dist_matrix: np.ndarray = distance_matrix(phrase_embeddings, phrase_embeddings, 2)  # create distance matrix
    # Cosine similarity analysis
    cosine_similarity_matrix: np.ndarray = cosine_matrix(phrase_embeddings)
    # tsne analysis
    tsne_embeddings: np.ndarray = generate_tsne_embeddings(phrase_embeddings)
    data["tsne-2d-x"] = tsne_embeddings[:, 0]
    data["tsne-2d-y"] = tsne_embeddings[:, 1]
    # visualize
    visualize_heatmap(
        euclid_dist_matrix,
        data["Topic"].unique(),
        "Distance matrix\n",
    )
    visualize_heatmap(
        cosine_similarity_matrix,
        data["Topic"].unique(),
        "Cosine similarity\n",
    )
    display_tsne(x_col="tsne-2d-x", y_col="tsne-2d-y", df=data)
    kmeans_cluster(
        x_col="tsne-2d-x",
        y_col="tsne-2d-y",
        embedding_col="Embedding",
        df=data,
        num_clusters=len(data["Topic"].unique()),
    )

    """
    Higher cosine score, more similar they are
    1 == perfect similarity ([1, 2, 3] and [1, 2, 3])
    0 == no similarity ([1, 0, 0] and [0, 1, 0])
    -1 ==  perfect dissimilarity ([-1, 0, 0] and [1, 0, 0])
    """
