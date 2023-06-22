import pandas as pd
import numpy as np
from chat import embed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine
import itertools


def cosine_similarity(vector_a: list, vector_b: list) -> float:
    # print(vector_a)
    # print(vector_b)
    return cosine(vector_a, vector_b)


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
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

    :param x_col:
    :param y_col:
    :param df:
    :return:
    """
    plt.figure(figsize=(6, 6))
    plt.title("T-SNE dimensionality reduction of Contrasting Topics Dataset")
    sns.scatterplot(
        x=x_col,
        y=y_col,
        hue="Topic",
        palette=sns.color_palette("hls", 5),
        data=df,
        legend="full",
        alpha=0.75
    )
    plt.show()
    plt.close()


def visualize_matrix(data) -> None:
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(data)
    # for i in range(data.shape[0] + 1):
    #     ax.axhline(i, color='white', lw=20)
    # for i in range(data.shape[1] + 1):
    #     ax.axvline(i, color='white', lw=60)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # preprocessing
    data: pd.DataFrame = pd.read_csv('contrast-dataset.csv', index_col='ID')
    data = one_hot_encoding(data)
    data['Embedding'] = data['Phrase'].apply(embed)
    phrase_embeddings: np.ndarray = np.array(list(data['Embedding'].values))

    euclid_dist_matrix: np.ndarray = distance_matrix(phrase_embeddings, phrase_embeddings, 2)  # create distance matrix

    visualize_matrix(euclid_dist_matrix)

    # tsne_embeddings: np.ndarray = generate_tsne_embeddings(phrase_embeddings)
    # data['tsne-2d-x'] = tsne_embeddings[:, 0]
    # data['tsne-2d-y'] = tsne_embeddings[:, 1]
    # display_tsne(x_col="tsne-2d-x", y_col="tsne-2d-y", df=data)

    averages: dict = {}
    for i in range(6):
        topic_embeddings = np.array(list(data.loc[data["Topic"] == i, 'Embedding'].values))
        avg: float = 0
        comparisons: int = 0
        for j, k in itertools.combinations(topic_embeddings, 2):
            avg += cosine_similarity(j, k)
            comparisons += 1
        averages[f'{i}'] = avg / comparisons
    print(averages)

    # print(cosine_similarity([0, 0, 1], [1, 0, 0]))  # 1
    # print(cosine_similarity([0, 0, 1], [1, 0, 1]))  # 0.29
    # print(cosine_similarity([0, 0, 1], [0, 0, 1]))  # 0
    """
    Lower cosine score, more similar they are
    """
