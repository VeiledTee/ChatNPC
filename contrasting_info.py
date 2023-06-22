import pandas as pd
import numpy as np
from chat import embed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


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
                n_iter=1000, n_jobs=-1).fit_transform(fit_on)


def display_tsne(x_col: str, y_col: str, df: pd.DataFrame) -> None:
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


# preprocessing
data: pd.DataFrame = pd.read_csv('contrast-dataset.csv', index_col='ID')
data = one_hot_encoding(data)
data['Embedding'] = data['Phrase'].apply(embed)
phrase_embeddings: np.ndarray = np.array(list(data['Embedding'].values))

tsne_embeddings: np.ndarray = generate_tsne_embeddings(phrase_embeddings)
data['tsne-2d-x'] = tsne_embeddings[:, 0]
data['tsne-2d-y'] = tsne_embeddings[:, 1]


if __name__ == '__main__':
    # print(tsne_embeddings.shape)
    # print(len(data['Embedding'].values))
    # print(data.head())
    display_tsne(x_col="tsne-2d-x", y_col="tsne-2d-y", df=data)
