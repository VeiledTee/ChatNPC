import pandas as pd
import numpy as np
from chat import embed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine
import itertools


def cosine_similarity(vector_a: list, vector_b: list) -> float:
    # print(vector_a)
    # print(vector_b)
    return 1 - cosine(vector_a, vector_b)


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode unique topic strings as integers
    :param df: Dataframe containing raw topics
    :return: Altered dataframe
    """
    mapping: dict = {}
    for index, topic in enumerate(df["Topic"].unique()):
        mapping[topic] = index
    print(mapping)
    return df.replace({'Topic': mapping})


def generate_tsne_embeddings(fit_on, dimensions: int = 2) -> np.ndarray:
    """
    Create embeddings of the specific dimension from data passed to the function
    :param fit_on: Data to fit T-SNE on
    :param dimensions: Num dimensions to reduce to
    :return: Generated embeddings
    """
    return TSNE(n_components=dimensions, learning_rate='auto',
                init='random', perplexity=30,
                n_iter=1000, n_jobs=-1, random_state=42).fit_transform(fit_on)


def display_tsne(x_col: str, y_col: str, df: pd.DataFrame) -> None:
    """
    Display T-SNE embedding data in scatterplot
    :param x_col: Name of column in df containing x-values of T-SNE embedding
    :param y_col: Name of column in df containing y-values of T-SNE embedding
    :param df: Dataframe object holding the T-SNE embeddings of original text
    :return: None
    """
    plt.figure(figsize=(6, 6))
    plt.title("T-SNE dimensionality reduction of Contrasting Topics Dataset")
    sns.scatterplot(
        x=x_col,
        y=y_col,
        hue="Topic",
        palette=sns.color_palette("hls", len(df["Topic"].unique())),
        data=df,
        legend="full",
        alpha=0.75
    )
    plt.show()
    plt.close()


def visualize_matrix(data_to_visualize, axis_labels, title: str, cosine_sim: bool = False) -> None:
    """
    Visualize a matrix ias a heatmap
    :param data_to_visualize: A 2d matrix representing the value of each cell in the final heatmap
    :param axis_labels: Labels for the y axis
    :param title: Title of the plot
    :param cosine_sim: True if representing cosine similarity values. Used to set the bounds of the colour bar. Default: False
    :return:
    """
    plt.figure(figsize=(8, 8))
    plt.title(title)
    ax = sns.heatmap(data_to_visualize, xticklabels=False, cmap="coolwarm")
    if cosine_sim:
        ax.collections[0].set_clim(-1, 1)
    else:
        ax.collections[0].set_clim(0, np.amax(data_to_visualize))
    ticks = []
    labels = []
    prev_label = None
    for i, label in enumerate(axis_labels):
        if label != prev_label:
            ticks.append(i)
            labels.append(label)
            prev_label = label
    ticks.append(i + 1)
    ax.yaxis.set_minor_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(axis='both', which='major', length=0)
    ax.tick_params(axis='y', which='minor', length=60)
    plt.tight_layout()
    plt.show()
    plt.close()


def average_similarity(df: pd.DataFrame, label_column: str = "Topic") -> dict:
    """
    Finds the average similarity of each topics phrases to each other
    :param df: Dataframe containing the embeddings and their labels
    :param label_column: Name of the column the labels are contained within
    :return: A dictionary stating the average cosine similarity score for each topic present in the dataset
    """
    averages: dict = {}
    for i in range(len(data[label_column].unique())):
        topic_embeddings = np.array(list(df.loc[data[label_column] == i, 'Embedding'].values))
        avg: float = 0
        comparisons: int = 0
        for j, k in itertools.combinations(topic_embeddings, 2):
            avg += cosine_similarity(j, k)
            comparisons += 1
        averages[f'{i}'] = avg / comparisons
    return averages


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    # print(len(vectors[0]))  # 768
    similarity_matrix: list = []
    for i in range(len(vectors)):
        curr: list = []
        for j in range(len(vectors)):
            curr.append(cosine_similarity(vectors[i], vectors[j]))
        similarity_matrix.append(curr)
    return np.array(similarity_matrix)


if __name__ == '__main__':
    # preprocessing
    data: pd.DataFrame = pd.read_csv('contrast-dataset.csv', index_col='ID')
    topic_labels: list = list(data['Topic'].values)
    data = one_hot_encoding(data)
    data['Embedding'] = data['Phrase'].apply(embed)
    phrase_embeddings: np.ndarray = np.array(list(data['Embedding'].values))
    # # Distance matrix analysis
    # euclid_dist_matrix: np.ndarray = distance_matrix(phrase_embeddings, phrase_embeddings, 2)  # create distance matrix
    # # Cosine similarity analysis
    # cosine_similarity_matrix: np.ndarray = cosine_matrix(phrase_embeddings)
    # tsne analysis
    tsne_embeddings: np.ndarray = generate_tsne_embeddings(phrase_embeddings)
    data['tsne-2d-x'] = tsne_embeddings[:, 0]
    data['tsne-2d-y'] = tsne_embeddings[:, 1]
    # visualize
    # visualize_matrix(euclid_dist_matrix, topic_labels, "Distance matrix generated through sentence embeddings from contrasting topic dataset", cosine_sim=False)
    # visualize_matrix(cosine_similarity_matrix, topic_labels, "Cosine similarity across sentence embeddings from contrasting topic dataset", cosine_sim=True)
    display_tsne(x_col="tsne-2d-x", y_col="tsne-2d-y", df=data)
    """
    Higher cosine score, more similar they are
    1 == perfect similarity ([1, 2, 3] and [1, 2, 3])
    0 == no similarity ([1, 0, 0] and [0, 1, 0])
    -1 ==  perfect dissimilarity ([-1, 0, 0] and [1, 0, 0])
    """
